import os
import pickle
import argparse
import torch
import glob
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import LongformerTokenizer

from task2x_regression_ensemble import (
    FileBasedSingleTaskDataset,
    SingleHeadLongformerModel, 
    ensemble_predict,
    mean_avg_error
)

def evaluate_ensemble_models(data_path, model_dir):
    """Evaluate ensemble of models"""
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_final_valid = data['X_final_valid']
    y_final_valid_century = data['y_final_valid_century']
    y_final_valid_decade = data['y_final_valid_decade']
    combined_path = data['combined_path']
    
    # Find model paths
    model_paths = []
    for fold_dir in sorted(glob.glob(os.path.join(model_dir, "fold_*"))):
        model_path = os.path.join(fold_dir, "best_model.pt")
        if os.path.exists(model_path):
            model_paths.append(model_path)
    
    if not model_paths:
        print(f"No models found in {model_dir}")
        return
    
    print(f"Found {len(model_paths)} models for ensemble evaluation")
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    # Create validation dataset with limited token length
    final_valid_dataset = FileBasedSingleTaskDataset(
        X_final_valid,
        lambda file: combined_path[file],
        y_final_valid_century,
        y_final_valid_decade,
        tokenizer,
        max_length=1536
    )
    
    final_valid_dataloader = DataLoader(final_valid_dataset, batch_size=16, shuffle=False)
    
    # Get true labels
    all_labels = []
    for batch in final_valid_dataloader:
        labels = batch['decade_labels'].numpy()
        all_labels.extend(labels)
    
    # Get predictions
    all_predictions = []
    
    for path in model_paths:
        # Load the model
        model = SingleHeadLongformerModel()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        
        batch_predictions = []
        with torch.no_grad():
            for batch in tqdm(final_valid_dataloader, desc=f"Predicting with model from {path}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                batch_predictions.extend(predictions.cpu().numpy())
        
        all_predictions.append(batch_predictions)
        
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    # Average predictions from all models
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Calculate metrics
    mae = mean_avg_error(np.array(all_labels), ensemble_predictions)
    mse = np.mean((np.array(all_labels) - ensemble_predictions)**2)
    final_rank = mae  # Final Rank is equivalent to MAE in this case
    
    print(f"Ensemble Final Results: MAE = {mae:.4f}, MSE = {mse:.4f}, Final Rank = {final_rank:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ensemble of models")
    parser.add_argument("--data", type=str, required=True, help="Path to prepared data pickle file")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing fold model directories")
    
    args = parser.parse_args()
    
    evaluate_ensemble_models(args.data, args.model_dir)
