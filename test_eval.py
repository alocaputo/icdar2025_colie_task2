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
from sklearn.model_selection import train_test_split
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add argument parser
parser = argparse.ArgumentParser(description='Evaluate models for ICDAR 2025')
parser.add_argument('--conformal', action='store_true', help='Use conformal prediction')
parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for conformal prediction (default: 0.1)')
parser.add_argument('--pred_method', type=str, default='argmax', 
                    choices=['argmax', 'set_argmax', 'threshold_max', 'threshold_max_norm', 'all'],
                    help='Method for selecting predictions (default: argmax, use "all" for all methods)')
parser.add_argument('--model_type', type=str, default='multitask', 
                    choices=['multitask', 'classification', 'consistent'],
                    help='Type of model to use: multitask (decade as 10 classes), classification (decade as 43 classes), or consistent (with temporal consistency)')
parser.add_argument('--model_path', type=str, default='models/task2x/best_model_notf.pt',
                    help='Path to the model file')
args = parser.parse_args()

class MultiTaskLongformerModel(nn.Module):
    def __init__(self):
        super(MultiTaskLongformerModel, self).__init__()
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        # Century head
        self.century_classifier = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )
        
        # Decade head
        self.decade_classifier = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        century_logits = self.century_classifier(pooled_output)
        decade_logits = self.decade_classifier(pooled_output)
        
        return century_logits, decade_logits

class MultiTaskClassificationModel(nn.Module):
    def __init__(self):
        super(MultiTaskClassificationModel, self).__init__()
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        # Century head (classification)
        self.century_classifier = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )
        
        # Decade head (43-class classification for combined century-decade values)
        self.decade_classifier = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 43)  # 43 classes for all possible decade values
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        century_logits = self.century_classifier(pooled_output)
        decade_logits = self.decade_classifier(pooled_output)
        
        return century_logits, decade_logits

class TestDataset(Dataset):
    def __init__(self, file_paths, path, tokenizer, max_length=1536, is_validation=False, df=None):
        self.file_paths = file_paths
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_validation = is_validation
        self.df = df
        
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
        
        # Add ground truth labels for validation data
        if self.is_validation and self.df is not None:
            # Extract the ID from filename (assuming format like valid_textXXX.txt)
            file_name = file_path
            encoding['century_label'] = self.df.loc[self.df['file_name'] == file_name, 'century'].values[0] - 1  # Convert to 0-indexed
            
        return encoding

def compute_conformal_scores(model, val_loader, device):
    """Compute conformal scores on validation data"""
    model.eval()
    # Change to store class-specific scores
    scores_by_class = {c: [] for c in range(5)}  # Assuming 5 classes for century
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing conformal scores"):
            true_label = batch.pop('century_label').cpu().numpy()
            file_id_batch = batch.pop('file_id')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            century_logits, _ = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            probas = F.softmax(century_logits, dim=1).cpu().numpy()
            
            # For each prediction, store 1 - probability of true class
            for i, true_idx in enumerate(true_label):
                scores_by_class[true_idx].append(1 - probas[i, true_idx])
            true_labels.extend(true_label)
    
    return scores_by_class, np.array(true_labels)

def apply_conformal_prediction(model, dataloader, device, thresholds, pred_method='argmax'):
    """Apply conformal prediction to test data using class-specific thresholds"""
    model.eval()
    file_ids = []
    prediction_sets = []
    point_predictions = []
    decade_predictions = []  # Add decade predictions list
    
    # Determine if we're using the classification model
    is_classification_model = isinstance(model, MultiTaskClassificationModel) or \
                             (hasattr(model, 'module') and isinstance(model.module, MultiTaskClassificationModel))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Applying conformal prediction"):
            file_id_batch = batch.pop('file_id')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            century_logits, decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            century_probas = F.softmax(century_logits, dim=1).cpu().numpy()
            decade_probas = F.softmax(decade_logits, dim=1).cpu().numpy()
            
            # For each prediction, compute prediction sets
            for i in range(century_probas.shape[0]):
                # Compute prediction set (classes with scores <= class-specific threshold)
                pred_set = []
                for c in range(century_probas.shape[1]):
                    if 1 - century_probas[i, c] <= thresholds[c]:
                        pred_set.append(c)  # Keep 0-indexed for now
                
                # If prediction set is empty, include all classes
                if not pred_set:
                    pred_set = list(range(century_probas.shape[1]))
                
                # Choose point prediction based on selected method
                if pred_method == 'argmax':
                    point_pred = np.argmax(century_probas[i])
                elif pred_method == 'set_argmax':
                    set_probas = [century_probas[i, c] for c in pred_set]
                    point_pred = pred_set[np.argmax(set_probas)]
                elif pred_method == 'threshold_max':
                    # Compare how much each class exceeds its threshold
                    margins = [(1 - century_probas[i, c] - thresholds[c]) for c in pred_set]
                    point_pred = pred_set[np.argmin(margins)]
                elif pred_method == 'threshold_max_norm':
                    scores = [(1 - century_probas[i, c]) / century_probas[i, c] for c in pred_set]
                    point_pred = pred_set[np.argmin(scores)]
                
                # Convert to 1-indexed for output
                point_pred += 1
                pred_set_1idx = [c + 1 for c in pred_set]
                
                # Get decade prediction based on model type
                if is_classification_model:
                    # For classification model, extract decade from combined class
                    combined_class = np.argmax(decade_probas[i])
                    decade_pred = (combined_class % 10) + 1
                else:
                    # Standard model with separate decade predictions
                    decade_pred = np.argmax(decade_probas[i]) + 1
                
                file_ids.append(file_id_batch[i])
                prediction_sets.append(pred_set_1idx)
                point_predictions.append(point_pred)
                decade_predictions.append(decade_pred)
    
    return file_ids, point_predictions, prediction_sets, decade_predictions

def generate_all_conformal_results(model, dataloader, device, thresholds):
    """Generate results for all prediction methods using class-specific thresholds"""
    model.eval()
    file_ids = []
    prediction_sets = []
    decade_predictions = []
    point_predictions = {
        'argmax': [],
        'set_argmax': [],
        'threshold_max': [],
        'threshold_max_norm': []
    }
    
    # Add counters and debug info
    set_sizes = []
    diff_from_argmax = {method: 0 for method in ['set_argmax', 'threshold_max', 'threshold_max_norm']}
    
    # Determine if we're using the classification model
    is_classification_model = isinstance(model, MultiTaskClassificationModel) or \
                             (hasattr(model, 'module') and isinstance(model.module, MultiTaskClassificationModel))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Applying conformal prediction for all methods"):
            file_id_batch = batch.pop('file_id')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            century_logits, decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            century_probas = F.softmax(century_logits, dim=1).cpu().numpy()
            decade_probas = F.softmax(decade_logits, dim=1).cpu().numpy()
            
            # For each prediction, compute prediction sets and all point predictions
            for i in range(century_probas.shape[0]):
                # Compute prediction set using class-specific thresholds
                pred_set = []
                for c in range(century_probas.shape[1]):
                    if 1 - century_probas[i, c] <= thresholds[c]:
                        pred_set.append(c)  # Keep 0-indexed for now
                
                # If prediction set is empty, include all classes
                if not pred_set:
                    pred_set = list(range(century_probas.shape[1]))
                
                # Track set size
                set_sizes.append(len(pred_set))
                
                # Standard argmax across all classes
                argmax_pred = np.argmax(century_probas[i]) + 1  # Convert to 1-indexed
                point_predictions['argmax'].append(argmax_pred)
                
                # Convert prediction set to 1-indexed for output
                pred_set_1idx = [c + 1 for c in pred_set]
                
                # Compute other methods
                # Set argmax
                set_probas = [century_probas[i, c] for c in pred_set]
                set_argmax_pred = pred_set_1idx[np.argmax(set_probas)]
                point_predictions['set_argmax'].append(set_argmax_pred)
                if set_argmax_pred != argmax_pred:
                    diff_from_argmax['set_argmax'] += 1
                
                # Threshold max - compares how much each class exceeds its threshold
                margins = [(1 - century_probas[i, c] - thresholds[c]) for c in pred_set]
                threshold_max_pred = pred_set_1idx[np.argmin(margins)]
                point_predictions['threshold_max'].append(threshold_max_pred)
                if threshold_max_pred != argmax_pred:
                    diff_from_argmax['threshold_max'] += 1
                
                # Threshold max norm
                scores_norm = [(1 - century_probas[i, c]) / century_probas[i, c] if century_probas[i, c] > 0 else float('inf') for c in pred_set]
                threshold_max_norm_pred = pred_set_1idx[np.argmin(scores_norm)]
                point_predictions['threshold_max_norm'].append(threshold_max_norm_pred)
                if threshold_max_norm_pred != argmax_pred:
                    diff_from_argmax['threshold_max_norm'] += 1
                
                # Add decade prediction based on model type
                if is_classification_model:
                    # For classification model, extract decade from combined class
                    combined_class = np.argmax(decade_probas[i])
                    decade_pred = (combined_class % 10) + 1
                else:
                    # Standard model with separate decade predictions
                    decade_pred = np.argmax(decade_probas[i]) + 1
                
                decade_predictions.append(decade_pred)
                
                file_ids.append(file_id_batch[i])
                prediction_sets.append(pred_set_1idx)
    
    # Print debug information
    print(f"\nPrediction set statistics:")
    print(f"Average set size: {np.mean(set_sizes):.2f}")
    print(f"Set size distribution: {np.bincount(set_sizes)}")
    print(f"Number of samples with different predictions from argmax:")
    for method, count in diff_from_argmax.items():
        print(f"  {method}: {count} ({count/len(file_ids)*100:.2f}%)")
    
    # Create a dictionary of results DataFrames for each method
    results = {}
    for method in point_predictions.keys():
        results[method] = pd.DataFrame({
            'id': file_ids,
            'century_label': point_predictions[method],
            'decade_label': decade_predictions
        })
    
    return results, file_ids, prediction_sets, decade_predictions

def run_inference(model, dataloader, device):
    model.eval()
    century_preds = []
    decade_preds = []
    file_ids = []
    
    # Determine if we're using the classification model
    is_classification_model = isinstance(model, MultiTaskClassificationModel) or \
                             (hasattr(model, 'module') and isinstance(model.module, MultiTaskClassificationModel))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            file_id_batch = batch.pop('file_id')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            century_logits, decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            # Get predicted classes (0-indexed)
            century_predictions = torch.argmax(century_logits, dim=1)
            
            if is_classification_model:
                # For classification model, extract decade from combined class
                combined_predictions = torch.argmax(decade_logits, dim=1)
                # The decade is the remainder when dividing by 10
                decade_predictions = combined_predictions % 10
            else:
                # Standard model with separate decade predictions
                decade_predictions = torch.argmax(decade_logits, dim=1)
            
            # Convert to actual century/decade (1-indexed for century)
            century_predictions = century_predictions.cpu().numpy() + 1
            decade_predictions = decade_predictions.cpu().numpy() + 1 
            
            century_preds.extend(century_predictions)
            decade_preds.extend(decade_predictions)
            file_ids.extend(file_id_batch)
    
    return file_ids, century_preds, decade_preds

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
    
    # Adjust batch size based on number of GPUs (optional)
    batch_size = 16 * max(1, num_gpus)  # Scale batch size with number of GPUs
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Using batch size: {batch_size}")
    
    # Load the model based on model_type argument
    print(f"Evaluating {args.model_type} model")
    model_path = args.model_path
    
    if args.model_type == 'classification':
        model = MultiTaskClassificationModel()
        output_file_base = './submissions/classification'
    elif args.model_type == 'consistent':
        # Import the consistent model from task2x_consistent.py
        try:
            from task2x_consistent import MultiTaskLongformerModel as ConsistentModel
            model = ConsistentModel()
            output_file_base = './submissions/consistent'
            print("Loaded consistent model architecture")
        except ImportError:
            print("Error: Could not import consistent model. Make sure task2x_consistent.py is in the correct location.")
            # Fall back to regular multitask model
            model = MultiTaskLongformerModel()
            output_file_base = './submissions/multitask'
    else:
        model = MultiTaskLongformerModel()
        output_file_base = './submissions/multitask'
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded {args.model_type} model from {model_path}")
    
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    if args.conformal:
        # Load validation data for conformal prediction
        print("Loading validation data for conformal prediction...")
        
        with open('blacklist.pkl', 'rb') as f:
            blacklist = pickle.load(f)
            
        blacklist_valid = blacklist['valid']

        # Load and prepare validation data
        valid21 = pd.read_csv('./data/Task2/task2.1/valid.csv')
        valid22 = pd.read_csv('./data/Task2/task2.2/valid.csv')

        # Preprocess data
        valid21.rename(columns={'label': 'century'}, inplace=True)
        valid21['file_name'] = valid21['id']
        valid21['id'] = valid21.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
        valid21.set_index('id', inplace=True)

        valid22.rename(columns={'label': 'century'}, inplace=True)
        valid22['file_name'] = valid22['id']
        valid22['id'] = valid22.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
        valid22.set_index('id', inplace=True)
        
        # Combine validation data
        valid_df = pd.concat([valid21, valid22])
        
        # Filter out IDs present in blacklist_valid
        print(f"Before filtering: {len(valid_df)} validation samples")
        valid_df = valid_df[~valid_df.index.isin(blacklist_valid)]
        print(f"After filtering blacklisted IDs: {len(valid_df)} validation samples")
        
        valid_files = valid_df['file_name'].tolist()

        # =============== NEW =============== 04/04
        
        # Filter out files containing 'gutenberg'
        gutenberg_free_files = []
        valid_path = './data/Task2/texts/valid'
        for file_name in valid_files:
            file_path = os.path.join(valid_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if 'gutenberg' not in content:
                        gutenberg_free_files.append(file_name)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
        
        print(f"After filtering Gutenberg files: {len(gutenberg_free_files)} validation samples")
        
        # Update valid_files list and valid_df
        valid_files = gutenberg_free_files
        valid_df = valid_df[valid_df['file_name'].isin(valid_files)]
        # =============== NEW =============== 04/04
        
        # Create validation dataset
        valid_path = './data/Task2/texts/valid'
        valid_dataset = TestDataset(valid_files, valid_path, tokenizer, is_validation=True, df=valid_df)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # Compute conformal scores - now class-specific
        print("Computing conformal scores...")
        scores_by_class, true_labels = compute_conformal_scores(model, valid_dataloader, device)
        
        # Compute threshold for each class
        alpha = args.alpha
        thresholds = {}
        print("Class-specific conformal thresholds:")
        for c in range(5):  # Assuming 5 classes for century
            if scores_by_class[c]:  # Make sure we have scores for this class
                thresholds[c] = np.quantile(scores_by_class[c], 1 - alpha)
                print(f"  Class {c+1}: {thresholds[c]:.4f} (based on {len(scores_by_class[c])} samples)")
            else:
                # If no examples of this class in validation set, use a default threshold
                thresholds[c] = 0.9  # Conservative default
                print(f"  Class {c+1}: {thresholds[c]:.4f} (default - no validation samples)")
        
        # Determine which methods to run
        if args.pred_method == 'all':
            # Run all methods in a single pass
            print("Running conformal prediction for all methods...")
            results_dict, _, _, _ = generate_all_conformal_results(model, test_dataloader, device, thresholds)
            
            # Extract model filename for the output CSV name
            model_filename = os.path.basename(args.model_path).replace('.pt', '')
            
            # Save results for each method to a separate file
            for method, results in results_dict.items():
                # Apply constraint: if century is 5, decade should not exceed 3
                results.loc[results['century_label'] == 5, 'decade_label'] = results.loc[results['century_label'] == 5, 'decade_label'].clip(upper=3)
                output_file = f"{output_file_base}_{model_filename}_conformal_alpha{alpha}_{method}.csv"
                results.to_csv(output_file, index=False)
                print(f"Saved {method} predictions to {output_file}")
            
            print("All methods processed and saved to separate files")
        else:
            # Run single method
            print(f"Running conformal prediction with method: {args.pred_method}...")
            results, _, _, _ = generate_all_conformal_results(model, test_dataloader, device, thresholds)
            
            # Extract model filename for the output CSV name
            model_filename = os.path.basename(args.model_path).replace('.pt', '')
            
            # Apply constraint: if century is 5, decade should not exceed 3
            results[args.pred_method].loc[results[args.pred_method]['century_label'] == 5, 'decade_label'] = \
                results[args.pred_method].loc[results[args.pred_method]['century_label'] == 5, 'decade_label'].clip(upper=3)
            
            # Save results to a CSV file
            output_file = f"{output_file_base}_{model_filename}_conformal_alpha{alpha}_{args.pred_method}.csv"
            results[args.pred_method].to_csv(output_file, index=False)
            print(f"Saved predictions to {output_file}")
        
    else:
        # Run regular inference
        file_ids, century_preds, decade_preds = run_inference(model, test_dataloader, device)
        
        # Extract model filename for the output CSV name
        model_filename = os.path.basename(args.model_path).replace('.pt', '')
        
        # Apply constraint: if century is 5, decade should not exceed 3
        for i in range(len(century_preds)):
            if century_preds[i] == 5 and decade_preds[i] > 3:
                decade_preds[i] = 3
        
        # Prepare a single submission file with both predictions
        results = pd.DataFrame({
            'id': file_ids,
            'century_label': century_preds,
            'decade_label': decade_preds
        })
        
        output_file = f"{output_file_base}_{model_filename}_argmax.csv"
    
    # Save results to a CSV file
    results.to_csv(output_file, index=False)
    
    print(f"Saved all predictions to {output_file}")