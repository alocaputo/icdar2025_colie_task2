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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Evaluate models for ICDAR 2025')
parser.add_argument('--model_type', type=str, default='regression', 
                    choices=['classification', 'regression'],
                    help='Type of model to use: classification or regression')
parser.add_argument('--model_path', type=str, default='models/best_model.pt',
                    help='Path to the model file')
parser.add_argument('--rounding', type=str, default='floor', 
                    choices=['round', 'floor'],
                    help='Method for converting regression output to integer (default: round)')
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

def run_inference(model, dataloader, device):
    model.eval()
    century_preds = []
    decade_preds = []
    file_ids = []
    
    # Check if we're using regression model
    is_regression_model = args.model_type == 'regression'
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            file_id_batch = batch.pop('file_id')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if is_regression_model:
                decade_prediction = model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask']
                )
                
                # Convert to integer based on rounding method
                if args.rounding == 'round':
                    absolute_predictions = torch.round(decade_prediction).cpu().numpy()
                else:  # floor
                    absolute_predictions = torch.floor(decade_prediction).cpu().numpy()
                
                century_predictions = (absolute_predictions // 10) + 1
                decade_predictions = (absolute_predictions % 10) + 1
            else:
                # Classification model, use argmax to get class indices
                decade_logits = model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask']
                )
                combined_predictions = torch.argmax(decade_logits, dim=1).cpu().numpy()
                
                century_predictions = (combined_predictions // 10) + 1
                decade_predictions = (combined_predictions % 10) + 1
            
            century_preds.extend(century_predictions)
            decade_preds.extend(decade_predictions)
            file_ids.extend(file_id_batch)
    
    return file_ids, century_preds, decade_preds

if __name__ == "__main__":

    test_path = './data/Task2/test'
    test_files = sorted(os.listdir(test_path))
    
    print(f"Test files found: {len(test_files)}")
    
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    test_dataset = TestDataset(test_files, test_path, tokenizer)
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    # Adjust batch size based on number of GPUs
    batch_size = 16 * max(1, num_gpus)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Using batch size: {batch_size}")
    
    # Load the model based on model_type argument
    print(f"Evaluating {args.model_type} model")
    model_path = args.model_path
    
    if args.model_type == 'classification':
        # Import and use the SingleHeadLongformerModel from new_task2x_combined_classification.py
        try:
            from new_task2x_combined_classification import SingleHeadLongformerModel
            model = SingleHeadLongformerModel()
            output_file_base = './test_submissions/classification'
            print("Loaded classification model architecture")
        except ImportError:
            print("Error: Could not import classification model. Make sure new_task2x_combined_classification.py is in the correct location.")
            exit(1)
    else:  # regression
        # Import and use the SingleHeadLongformerModel from new_task2x_combined_regression.py
        try:
            from task2x_combined_regression import SingleHeadLongformerModel
            model = SingleHeadLongformerModel()
            output_file_base = './test_submissions/regression'
            print("Loaded regression model architecture")
        except ImportError:
            print("Error: Could not import regression model. Make sure new_task2x_combined_regression.py is in the correct location.")
            exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded {args.model_type} model from {model_path}")
    
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    file_ids, century_preds, decade_preds = run_inference(model, test_dataloader, device)
    
    model_filename = os.path.basename(args.model_path).replace('.pt', '')
    
    # Apply thresholding for predictions
    for i in range(len(century_preds)):
        if century_preds[i] == 5 and decade_preds[i] > 3:
            decade_preds[i] = 3
    
    results = pd.DataFrame({
        'id': file_ids,
        'century_label': century_preds,
        'decade_label': decade_preds
    })
    
    if args.model_type == 'regression':
        output_file = f"{output_file_base}_{model_filename}_{args.rounding}.csv"
    else:
        output_file = f"{output_file_base}_{model_filename}.csv"
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results.to_csv(output_file, index=False)
    
    print(f"Saved all predictions to {output_file}")