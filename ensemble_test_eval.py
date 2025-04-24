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
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Evaluate ensemble models for ICDAR 2025')
parser.add_argument('--model_path', type=str, default='models/',
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
        
        # Regression head
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
            
            # Get predictions all the predictions
            model_predictions = []
            for model in models:
                preds = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                model_predictions.append(preds.cpu().numpy())
            
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
    
    return file_ids, all_predictions

def find_best_ensemble_config():
    """
    Compute the best ensemble configuration by evaluating all models and their combinations.
    Save the best ensemble configuration to a pickle file.
    """
    print("Computing best ensemble configuration...")
    
    # Load validation data
    texts_path = './old_data/Task2/texts'
    valid_path = os.path.join(texts_path, 'valid')
    
    valid21 = pd.read_csv('./old_data/Task2/task2.1/valid.csv')
    valid22 = pd.read_csv('./old_data/Task2/task2.2/valid.csv')
    
    valid21.rename(columns={'label': 'century'}, inplace=True)
    valid21['file_name'] = valid21['id']
    valid21['id'] = valid21.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
    valid21.set_index('id', inplace=True)
    
    valid22.rename(columns={'label': 'century'}, inplace=True)
    valid22['file_name'] = valid22['id']
    valid22['id'] = valid22.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
    valid22.set_index('id', inplace=True)
    
    def convert_to_absolute_decade(century, decade):
        """Convert century (0-4) and decade (0-9) to absolute decade (0-42)"""
        return century * 10 + decade
    
    def mean_avg_error(y_true, y_pred):
        """Calculate Mean Absolute Error between true and predicted values"""
        return np.mean(np.abs(y_true - y_pred))
    
    X_valid_21, y_valid_21 = [], []
    X_valid_22, y_valid_22 = [], []
    
    for idx, row in valid21.iterrows():
        file_name = row.file_name
        century = row.century
        
        try:
            with open(os.path.join(valid_path, file_name), 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                
            if 'gutenberg' in text.lower() and 'project' in text.lower():
                continue
                
            X_valid_21.append(file_name)
            y_valid_21.append(century-1)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    
    for idx, row in valid22.iterrows():
        file_name = row.file_name
        century = row.century
        
        try:
            with open(os.path.join(valid_path, file_name), 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                
            if 'gutenberg' in text.lower() and 'project' in text.lower():
                continue
                
            X_valid_22.append(file_name)
            y_valid_22.append(century-1)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    
    print(f"Validation samples loaded: {len(X_valid_21)}")
    
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    class FileBasedSingleTaskDataset(Dataset):
        def __init__(self, file_paths, path, century_labels, decade_labels, tokenizer, max_length=1536):
            self.file_paths = file_paths
            self.path = path
            self.century_labels = century_labels
            self.decade_labels = decade_labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.file_paths)
        
        def __getitem__(self, idx):
            file_path = self.file_paths[idx]
            century = self.century_labels[idx]
            decade = self.decade_labels[idx]
            
            # Convert to absolute decade label
            absolute_decade = convert_to_absolute_decade(century, decade)
            
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
            encoding['decade_labels'] = torch.tensor(absolute_decade, dtype=torch.long)
            
            return encoding
    
    valid_dataset = FileBasedSingleTaskDataset(X_valid_21, valid_path, y_valid_21, y_valid_22, tokenizer)
    batch_size = 16 * max(1, torch.cuda.device_count())
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Find all model checkpoint files
    model_dir = args.model_path
    model_files = glob.glob(os.path.join(model_dir, 'best_single_head_regression_model_epoch_*.pt'))
    print(f"Found {len(model_files)} model checkpoints")
    
    # Load models
    models = []
    model_class = SingleHeadLongformerModel
    
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append({
            'name': model_name,
            'model': model
        })
        print(f"Loaded {model_name}")
    
    # Evaluate each model
    model_results = []
    
    for model_info in models:
        model = model_info['model']
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Evaluating {model_info['name']}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                decade_labels = batch.pop('decade_labels').float().to(device)
                
                predictions = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(decade_labels.cpu().numpy())
        
        # Calculate metrics
        predictions_array = np.array(all_predictions)
        labels_array = np.array(all_labels)
        
        mae = mean_avg_error(labels_array, predictions_array)
        mse = np.mean((labels_array - predictions_array)**2)
        
        model_results.append({
            'name': model_info['name'],
            'predictions': predictions_array,
            'mae': mae,
            'mse': mse
        })
        print(f"{model_info['name']}: MAE = {mae:.4f}, MSE = {mse:.4f}")
    
    # Get validation labels
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            all_labels.extend(batch['decade_labels'].float().cpu().numpy())
    labels_array = np.array(all_labels)
    
    # Simple average ensemble
    predictions_matrix = np.vstack([result['predictions'] for result in model_results])
    ensemble_predictions = np.mean(predictions_matrix, axis=0)
    
    # Calculate ensemble metrics
    ensemble_mae = mean_avg_error(labels_array, ensemble_predictions)
    ensemble_mse = np.mean((labels_array - ensemble_predictions)**2)
    print(f"Simple Average Ensemble: MAE = {ensemble_mae:.4f}, MSE = {ensemble_mse:.4f}")
    
    # Weighted ensemble (inverse of MAE as weight)
    weights = 1 / np.array([result['mae'] for result in model_results])
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    # Weighted ensemble predictions
    weighted_ensemble_predictions = np.zeros_like(ensemble_predictions)
    for i, result in enumerate(model_results):
        weighted_ensemble_predictions += weights[i] * result['predictions']
    
    # Calculate weighted ensemble metrics
    weighted_mae = mean_avg_error(labels_array, weighted_ensemble_predictions)
    weighted_mse = np.mean((labels_array - weighted_ensemble_predictions)**2)
    print(f"Weighted Ensemble: MAE = {weighted_mae:.4f}, MSE = {weighted_mse:.4f}")
    
    best_ensemble = {
        'type': 'weighted' if weighted_mae < ensemble_mae else 'average',
        'models': [os.path.basename(m['name']) for m in models],
        'weights': weights.tolist() if weighted_mae < ensemble_mae else None,
        'mae': min(weighted_mae, ensemble_mae),
        'mse': weighted_mse if weighted_mae < ensemble_mae else ensemble_mse
    }
    
    with open(args.ensemble_info, 'wb') as f:
        pickle.dump(best_ensemble, f)
    
    print(f"Saved best ensemble configuration (type: {best_ensemble['type']}) to {args.ensemble_info}")
    return best_ensemble

if __name__ == "__main__":

    # Load testset
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
    
    print(f"Evaluating ensemble of regression models")
    
    # Load ensemble configuration, if available otherwhise compute it
    if not os.path.exists(args.ensemble_info):
        print(f"Ensemble configuration file {args.ensemble_info} not found.")
        find_best_ensemble_config()

    with open(args.ensemble_info, 'rb') as f:
        ensemble_config = pickle.load(f)
    
    print(f"Loaded ensemble configuration with {len(ensemble_config['models'])} models")
    
    # Get paths to all models in the ensemble
    model_dir = args.model_path
    models = []
    
    model_class = SingleHeadLongformerModel
    output_file_base = './test_submissions/regression_ensemble'
    print("Loaded regression model architecture")
    
    # Load all models in the ensemble
    for model_file in ensemble_config['models']:
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found, using {model_file} directly")
            model_path = model_file
        
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        if num_gpus > 1:
            model = nn.DataParallel(model)
            
        model.to(device)
        models.append(model)
        print(f"Loaded model: {model_file}")


    for ensemble_type in ['weighted']: #['average', 'weighted']:
        print(f"\nRunning {ensemble_type} ensemble...")
        
        # Set weights based on ensemble type
        weights = ensemble_config['weights'] if ensemble_type == 'weighted' and 'weights' in ensemble_config else None
        
        file_ids, raw_predictions = run_ensemble_inference(models, weights, test_dataloader, device)
        
        # Process with both rounding methods
        rounding_methods = ['round', 'floor']
        
        for rounding_method in rounding_methods:
            if rounding_method == 'round':
                absolute_predictions = np.round(raw_predictions).astype(int)
            else:  # floor
                absolute_predictions = np.floor(raw_predictions).astype(int)
            
            # Convert absolute decade to century and decade
            century_preds = (absolute_predictions // 10) + 1
            decade_preds = (absolute_predictions % 10) + 1
            
            # Apply a teshold to ensure valid predictions (should not be necessary)
            for i in range(len(century_preds)):
                if century_preds[i] == 5 and decade_preds[i] > 3:
                    decade_preds[i] = 3
            
            results = pd.DataFrame({
                'id': file_ids,
                'century_label': century_preds,
                'decade_label': decade_preds
            })
            
            output_file = f"{output_file_base}_{ensemble_type}_{args.ensemble_info}_{rounding_method}.csv"  
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results.to_csv(output_file, index=False)
            
            print(f"Saved {ensemble_type} ensemble with {rounding_method} rounding to {output_file}")
