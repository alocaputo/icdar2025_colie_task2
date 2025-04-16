import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel, LongformerTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import argparse
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add argument parser - simplified to only essential arguments
parser = argparse.ArgumentParser(description='Evaluate ensemble models for ICDAR 2025')
parser.add_argument('--model_path', type=str, default='models/task2x',
                    help='Path to the directory containing model files')
parser.add_argument('--ensemble_info', type=str, default='best_ensemble_info.pkl',
                    help='Path to ensemble configuration file')
args = parser.parse_args()

class TestDataset(Dataset):
    def __init__(self, file_paths, path, tokenizer, max_length=1536):
        self.file_paths = file_paths
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_id = file_path  # Keep the original filename for reference
        
        with open(os.path.join(self.path, file_path), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
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

# Model definition
class SingleHeadLongformerModel(nn.Module):
    def __init__(self):
        super(SingleHeadLongformerModel, self).__init__()
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        # Regression head - outputs a single continuous value
        self.decade_regressor = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        decade_prediction = self.decade_regressor(pooled_output).squeeze(-1)
        
        return decade_prediction
    
def run_ensemble_inference(models, weights, dataloader, device):
    """Run inference with an ensemble of models"""
    for model in models:
        model.eval()
        
    all_predictions = []
    file_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running ensemble inference"):
            file_id_batch = batch.pop('file_id')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get predictions from each model
            model_predictions = []
            for model in models:
                preds = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                # Apply floor to each model's predictions before combining
                preds = torch.floor(preds)
                model_predictions.append(preds.cpu().numpy())
            
            # Stack predictions into a matrix
            predictions_matrix = np.vstack([p[np.newaxis, :] for p in model_predictions])
            
            # Apply weights if provided, otherwise use simple average
            if weights is not None:
                batch_preds = np.zeros_like(model_predictions[0])
                for i, preds in enumerate(model_predictions):
                    batch_preds += weights[i] * preds
            else:
                batch_preds = np.mean(predictions_matrix, axis=0)
                
            all_predictions.extend(batch_preds)
            file_ids.extend(file_id_batch)
    
    # Return raw predictions for multiple rounding methods
    return file_ids, all_predictions

if __name__ == "__main__":
    # Path to test files
    test_path = './data/Task2/test'
    test_files = sorted(os.listdir(test_path))
    
    print(f"Test files found: {len(test_files)}")
    
    # Initialize tokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    # Create test dataset
    test_dataset = TestDataset(test_files, test_path, tokenizer)
    
    # Count available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    # Adjust batch size based on number of GPUs
    batch_size = 16 * max(1, num_gpus)  # Scale batch size with number of GPUs
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Using batch size: {batch_size}")
    
    print(f"Evaluating ensemble of regression models")
    
    # Load ensemble configuration
    with open(args.ensemble_info, 'rb') as f:
        ensemble_config = pickle.load(f)
    
    print(f"Loaded ensemble configuration with {len(ensemble_config['models'])} models")
    
    # Get paths to all models in the ensemble
    model_dir = args.model_path
    models = []
    
    # Import model architecture
    model_class = SingleHeadLongformerModel
    output_file_base = './new_submissions2/regression_ensemble_new'
    print("Loaded regression model architecture")
    
    # Load all models in the ensemble
    for model_file in ensemble_config['models'][:-1]:
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found, using {model_file} directly")
            model_path = model_file
        
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Wrap model with DataParallel if multiple GPUs are available
        if num_gpus > 1:
            model = nn.DataParallel(model)
            
        model.to(device)
        models.append(model)
        print(f"Loaded model: {model_file}")
    
    # Process both ensemble types
    ensemble_types = ['average', 'weighted']
    
    for ensemble_type in ensemble_types:
        print(f"\nRunning {ensemble_type} ensemble...")
        
        # Set weights based on ensemble type
        weights = ensemble_config['weights'] if ensemble_type == 'weighted' and 'weights' in ensemble_config else None
        
        # Run ensemble inference to get raw predictions
        file_ids, raw_predictions = run_ensemble_inference(models, weights, test_dataloader, device)
        
        # Process with both rounding methods
        rounding_methods = ['round', 'floor']
        
        for rounding_method in rounding_methods:
            # Apply rounding method
            if rounding_method == 'round':
                absolute_predictions = np.round(raw_predictions).astype(int)
            else:  # floor
                absolute_predictions = np.floor(raw_predictions).astype(int)
            
            # Convert absolute decade to century and decade
            century_preds = (absolute_predictions // 10) + 1
            decade_preds = (absolute_predictions % 10) + 1
            
            # Apply constraint: if century is 5, decade should not exceed 3
            for i in range(len(century_preds)):
                if century_preds[i] == 5 and decade_preds[i] > 3:
                    decade_preds[i] = 3
            
            # Prepare a submission file with predictions
            results = pd.DataFrame({
                'id': file_ids,
                'century_label': century_preds,
                'decade_label': decade_preds
            })
            
            # Generate output filename
            output_file = f"{output_file_base}_{ensemble_type}_{args.ensemble_info}_{rounding_method}.csv"
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save results to a CSV file
            results.to_csv(output_file, index=False)
            
            print(f"Saved {ensemble_type} ensemble with {rounding_method} rounding to {output_file}")
