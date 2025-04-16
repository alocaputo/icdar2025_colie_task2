import torch
import torch.nn as nn
from transformers import LongformerTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import argparse
import glob
import json

# Import the model class
from new_task2x_combined_classification_montecarlo import SingleHeadLongformerModel

# Set device - will be modified based on arguments
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate ensemble of Monte Carlo models')
    
    parser.add_argument('--models_dir', type=str, default='new_models/montecarlo_classifiers',
                       help='Directory containing Monte Carlo model folders')
    
    parser.add_argument('--ensemble_method', type=str, default='all',
                       choices=['hard_vote', 'soft_vote', 'weighted', 'all'],
                       help='Method to combine predictions (default: all)')
    
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='Use multiple GPUs if available')
    
    return parser.parse_args()

class TestDataset(Dataset):
    def __init__(self, file_paths, path, tokenizer, max_length=1536):
        self.file_paths = file_paths
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-load all texts to avoid reopening files
        self.texts = []
        print("Loading test texts...")
        for file_path in tqdm(file_paths):
            with open(os.path.join(path, file_path), 'r', encoding='utf-8', errors='replace') as f:
                self.texts.append(f.read())
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_id = file_path  # Keep the original filename for reference
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['file_id'] = file_id
        
        return encoding

def load_models(models_dir, use_multi_gpu=False):
    """Load all Monte Carlo models from the specified directory"""
    models = []
    model_weights = []
    model_dirs = sorted(glob.glob(os.path.join(models_dir, "model_*")))
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Check if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    if use_multi_gpu and num_gpus > 1:
        print(f"Using {num_gpus} GPUs for inference")
        multi_gpu = True
    else:
        if use_multi_gpu and num_gpus <= 1:
            print("Multi-GPU requested but only one GPU found. Using single GPU.")
        multi_gpu = False
    
    for model_dir in model_dirs:
        best_model_path = os.path.join(model_dir, "best_model.pt")
        metrics_path = os.path.join(model_dir, "metrics.json")
        
        # Check if best_model.pt exists
        if os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            # Look for epoch model files (model_epoch_*.pt)
            epoch_model_files = glob.glob(os.path.join(model_dir, "model_epoch_*.pt"))
            if not epoch_model_files:
                print(f"No model files found in {model_dir}, skipping...")
                continue
            
            # Parse epoch numbers and get the highest one
            epoch_nums = []
            for f in epoch_model_files:
                try:
                    epoch_num = int(os.path.basename(f).replace("model_epoch_", "").replace(".pt", ""))
                    epoch_nums.append((epoch_num, f))
                except ValueError:
                    continue
            
            if not epoch_nums:
                print(f"No valid epoch model files found in {model_dir}, skipping...")
                continue
            
            # Sort by epoch number (descending) and get the path to the highest epoch
            epoch_nums.sort(reverse=True)
            highest_epoch, model_path = epoch_nums[0]
            print(f"Using highest epoch model ({highest_epoch}) in {model_dir}")
        
        # Load model metrics if available
        weight = 1.0  # Default weight
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Use inverse of MAE as weight (lower MAE = higher weight)
                mae = metrics.get("mae", 1.0)
                weight = 1.0 / max(mae, 0.001)  # Prevent division by zero
                print(f"Loaded model from {model_dir} with MAE: {mae:.4f}, weight: {weight:.4f}")
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Could not read metrics from {metrics_path}, using default weight")
        else:
            print(f"No metrics file found in {model_dir}, using default weight")
        
        # Load the model
        try:
            model = SingleHeadLongformerModel()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            # Wrap model in DataParallel if using multiple GPUs
            if multi_gpu:
                model = nn.DataParallel(model)
                
            model.eval()
            models.append(model)
            model_weights.append(weight)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            continue
    
    # Normalize weights to sum to 1
    if model_weights:
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()
    
    print(f"Successfully loaded {len(models)} models")
    
    return models, model_weights

def run_ensemble_inference(models, model_weights, dataloader, device):
    """Run inference using ensemble of models and return results for all ensemble methods"""
    file_ids = []
    all_logits = []
    
    print("Running inference with all models...")
    
    # Get logits from all models
    for i, model in enumerate(models):
        model_logits = []
        model.eval()
        
        print(f"Running inference with model {i+1}/{len(models)}")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if i == 0:  # Only save file IDs once
                    file_ids.extend(batch.pop('file_id'))
                else:
                    _ = batch.pop('file_id')
                    
                batch = {k: v.to(device) for k, v in batch.items()}
                
                logits = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                model_logits.append(logits.cpu())
        
        # Concatenate all batch predictions
        model_logits = torch.cat(model_logits, dim=0)
        all_logits.append(model_logits)
    
    # Results dictionary to store predictions for all methods
    results = {}
    
    # Hard vote ensemble
    all_preds = [torch.argmax(logits, dim=1).numpy() for logits in all_logits]
    hard_vote_preds = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=np.vstack(all_preds)
    )
    results['hard_vote'] = hard_vote_preds
    
    # Soft vote ensemble
    all_probs = [torch.softmax(logits, dim=1).numpy() for logits in all_logits]
    avg_probs = np.mean(np.stack(all_probs), axis=0)
    soft_vote_preds = np.argmax(avg_probs, axis=1)
    results['soft_vote'] = soft_vote_preds
    
    # Weighted ensemble
    weighted_probs = np.zeros_like(all_probs[0])
    for i, probs in enumerate(all_probs):
        weighted_probs += probs * model_weights[i]
    weighted_preds = np.argmax(weighted_probs, axis=1)
    results['weighted'] = weighted_preds
    
    return file_ids, results

def process_predictions(predictions):
    """Convert raw predictions to century and decade format"""
    # Convert predictions to century and decade
    century_preds = (predictions // 10) + 1
    decade_preds = (predictions % 10) + 1
    
    # Apply constraint: if century is 5, decade should not exceed 3
    for i in range(len(century_preds)):
        if century_preds[i] == 5 and decade_preds[i] > 3:
            decade_preds[i] = 3
    
    return century_preds, decade_preds

def main():
    args = parse_arguments()
    
    # Fixed batch size of 16
    batch_size = 16
    
    # If using multiple GPUs, increase batch size proportionally
    if args.use_multi_gpu:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            batch_size = batch_size * num_gpus
            print(f"Multi-GPU enabled: Increasing batch size to {batch_size}")
    
    # Create output directory
    output_dir = "new_submissions/montecarlo_class"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to test files
    test_path = './data/Task2/test'
    test_files = sorted(os.listdir(test_path))
    
    print(f"Test files found: {len(test_files)}")
    
    # Initialize tokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    # Create test dataset
    test_dataset = TestDataset(test_files, test_path, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Using batch size: {batch_size}")
    
    # Load all models
    models, model_weights = load_models(args.models_dir, args.use_multi_gpu)
    
    if not models:
        print("No models found. Exiting.")
        return
    
    # Run ensemble inference for all methods
    file_ids, ensemble_results = run_ensemble_inference(
        models, model_weights, test_dataloader, device
    )
    
    # Get timestamp for filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process and save results for each requested ensemble method
    methods_to_save = list(ensemble_results.keys()) if args.ensemble_method == 'all' else [args.ensemble_method]
    
    for method in methods_to_save:
        # Process predictions
        century_preds, decade_preds = process_predictions(ensemble_results[method])
        
        # Prepare results DataFrame
        results_df = pd.DataFrame({
            'id': file_ids,
            'century_label': century_preds,
            'decade_label': decade_preds
        })
        
        # Create output file path
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        output_file = os.path.join(method_dir, f"ensemble_{method}_{timestamp}.csv")
        
        # Save results
        results_df.to_csv(output_file, index=False)
        print(f"Saved {method} ensemble predictions to {output_file}")

if __name__ == "__main__":
    main()
