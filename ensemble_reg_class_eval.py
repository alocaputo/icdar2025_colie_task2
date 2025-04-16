import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from transformers import LongformerModel, LongformerTokenizer
from torch.utils.data import Dataset, DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
OUTPUT_DIR = './new_submissions/regression_ensemble_reg_class/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardcoded batch size
BATCH_SIZE = 16

# Parse arguments - added data_base_path argument
parser = argparse.ArgumentParser(description='Evaluate ensemble of regression and classifier models with conformal prediction')
parser.add_argument('--regression_model_path', type=str, default='old_models/task2x/best_single_head_regression_model.pt',
                    help='Path to the regression model')
parser.add_argument('--classifier_model_path', type=str, default='old_models/task2x/best_single_head_model.pt',
                    help='Path to the classifier model')
parser.add_argument('--data_base_path', type=str, default='./old_data/data',
                    help='Base path for data directory')
parser.add_argument('--confidence_level', type=float, default=0.9,
                    help='Confidence level for conformal prediction intervals')
args = parser.parse_args()

# Define paths based on data_base_path
TEST_PATH = './data/Task2/test'  # Test path is separate
VALID_PATH = os.path.join(args.data_base_path, 'Task2/texts/valid')
VALID_CSV = os.path.join(args.data_base_path, 'Task2/task2.1/valid.csv')

if not os.path.exists(VALID_PATH):
    VALID_PATH = os.path.join(args.data_base_path, 'Task2/texts/')

# Model definitions
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        
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

class ClassifierModel(nn.Module):
    def __init__(self, num_decades=43):
        super(ClassifierModel, self).__init__()
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        # Single decade classifier head
        self.decade_classifier = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_decades)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        decade_logits = self.decade_classifier(pooled_output)
        
        return decade_logits

class FileBasedDataset(Dataset):
    def __init__(self, file_paths, path, labels=None, tokenizer=None, max_length=1536):
        self.file_paths = file_paths
        self.path = path
        self.labels = labels
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
        
        if self.labels is not None:
            encoding['decade_labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return encoding

def get_regression_predictions(model, dataloader):
    model.eval()
    all_preds = []
    all_file_ids = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting regression predictions"):
            file_ids = batch.pop('file_id')
            labels = batch.pop('decade_labels', None)
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            all_preds.extend(preds.cpu().numpy())
            all_file_ids.extend(file_ids)
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_preds), all_file_ids, np.array(all_labels) if all_labels else None

def get_classifier_predictions(model, dataloader):
    model.eval()
    all_probs = []
    all_preds = []
    all_file_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting classifier predictions"):
            file_ids = batch.pop('file_id')
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_file_ids.extend(file_ids)
            
    return np.array(all_probs), np.array(all_preds), all_file_ids

# Function to compute prediction intervals based on desired confidence level
def get_prediction_intervals(pred, nonconformity_scores, confidence_level=0.9):
    # Sort nonconformity scores
    sorted_scores = np.sort(nonconformity_scores)
    
    # Find the threshold for the given confidence level
    n = len(sorted_scores)
    # Add 1 to ensure valid index and adjust confidence level to be conservative
    index = int(np.ceil((n+1) * confidence_level)) - 1
    index = min(index, n-1)  # Ensure index is valid
    threshold = sorted_scores[index]
    
    # Calculate prediction intervals
    lower_bound = np.floor(pred - threshold).astype(int)
    upper_bound = np.ceil(pred + threshold).astype(int)
    
    # Ensure values are within valid range (0 to 42)
    lower_bound = np.maximum(lower_bound, 0)
    upper_bound = np.minimum(upper_bound, 42)
    
    return lower_bound, upper_bound

def ensemble_prediction(reg_pred, lower_bound, upper_bound, class_probs):
    # Create an interval of integers from lower_bound to upper_bound
    interval = np.arange(lower_bound, upper_bound + 1)
    
    # Filter probabilities within the interval
    filtered_probs = np.array([class_probs[i] if i in interval else 0 for i in range(len(class_probs))])
    
    # If no valid probabilities in interval, return the regression prediction rounded to nearest integer
    if np.sum(filtered_probs) == 0:
        return int(np.round(reg_pred))
    
    # Otherwise, return the class with the highest probability within the interval
    return np.argmax(filtered_probs)

def compute_nonconformity_scores(predictions, labels):
    # Use absolute error as nonconformity score for regression
    return np.abs(predictions - labels)

def load_validation_data():
    """Load and prepare validation data from files"""
    print("Loading validation data...")
    
    # Read validation CSV
    valid_df = pd.read_csv(VALID_CSV)
    valid_df.rename(columns={'label': 'century'}, inplace=True)
    valid_df['file_name'] = valid_df['id']
    valid_df['id'] = valid_df.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
    valid_df.set_index('id', inplace=True)
    
    # Filter out problematic files (like in the notebook)
    file_paths = []
    labels = []
    blacklist_valid = []  # If you have a blacklist, use it here
    
    # Process validation data
    for idx, row in valid_df.iterrows():
        if idx in blacklist_valid:
            continue
            
        file_name = row.file_name
        century = row.century
        decade = 0  # Assuming decade is 0, adjust if you have decade information
        
        # Check for gutenberg text (optional)
        with open(os.path.join(VALID_PATH, file_name), 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
            if 'gutenberg' in text.lower():
                continue
        
        # Calculate absolute decade (0-42)
        absolute_decade = (century - 1) * 10 + decade
        
        file_paths.append(file_name)
        labels.append(absolute_decade)
    
    print(f"Loaded {len(file_paths)} validation samples")
    return file_paths, labels

def save_model_predictions(file_ids, predictions, output_file, model_type=""):
    """
    Save individual model predictions to a CSV file
    
    Args:
        file_ids: List of file identifiers
        predictions: Array of model predictions
        output_file: Path to save the predictions
        model_type: Type of model (regressor or classifier)
    """
    if model_type == "regressor":
        # For regression model, save both raw and rounded predictions
        raw_preds = predictions
        rounded_preds = np.round(predictions).astype(int)
        
        # Convert to century and decade
        century_preds = (rounded_preds // 10) + 1
        decade_preds = (rounded_preds % 10) + 1
        
        # Apply constraint: if century is 5, decade should not exceed 3
        for i in range(len(century_preds)):
            if century_preds[i] == 5 and decade_preds[i] > 3:
                decade_preds[i] = 3
                
        results = pd.DataFrame({
            'id': file_ids,
            'raw_prediction': raw_preds,
            'rounded_prediction': rounded_preds,
            'century_label': century_preds,
            'decade_label': decade_preds
        })
    
    elif model_type == "classifier":
        # For classifier model, save class predictions directly
        century_preds = (predictions // 10) + 1
        decade_preds = (predictions % 10) + 1
        
        # Apply constraint: if century is 5, decade should not exceed 3
        for i in range(len(century_preds)):
            if century_preds[i] == 5 and decade_preds[i] > 3:
                decade_preds[i] = 3
                
        results = pd.DataFrame({
            'id': file_ids,
            'absolute_prediction': predictions,
            'century_label': century_preds,
            'decade_label': decade_preds
        })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"Saved {model_type} predictions to {output_file}")

def main():
    print(f"Using device: {device}")
    print(f"Models: \n  Regression: {args.regression_model_path}\n  Classifier: {args.classifier_model_path}")
    print(f"Data paths: \n  Base: {args.data_base_path}\n  Test: {TEST_PATH}\n  Validation: {VALID_PATH}\n  Validation CSV: {VALID_CSV}")
    
    # Initialize tokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    # Load models
    print("Loading regression model...")
    regression_model = RegressionModel()
    regression_model.load_state_dict(torch.load(args.regression_model_path, map_location=device))
    regression_model.to(device)
    
    print("Loading classifier model...")
    classifier_model = ClassifierModel()
    classifier_model.load_state_dict(torch.load(args.classifier_model_path, map_location=device))
    classifier_model.to(device)
    
    # Load validation data and compute nonconformity scores
    valid_files, valid_labels = load_validation_data()
    valid_dataset = FileBasedDataset(valid_files, VALID_PATH, valid_labels, tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get predictions for validation set
    print("Getting predictions for validation set to compute nonconformity scores...")
    valid_reg_preds, _, valid_labels_array = get_regression_predictions(regression_model, valid_dataloader)
    
    # Compute nonconformity scores
    print("Computing nonconformity scores on the validation set...")
    nonconformity_scores = compute_nonconformity_scores(valid_reg_preds, valid_labels_array)
    
    # Save nonconformity scores
    print(f"Saving nonconformity scores to {OUTPUT_DIR}/nonconformity_scores.pkl...")
    with open(f"{OUTPUT_DIR}/nonconformity_scores.pkl", 'wb') as f:
        pickle.dump(nonconformity_scores, f)
    
    # Now process test data
    test_files = sorted(os.listdir(TEST_PATH))
    print(f"Found {len(test_files)} test files")
    
    # Create test dataset and dataloader
    test_dataset = FileBasedDataset(test_files, TEST_PATH, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get predictions for the test set
    print("Getting predictions for test set...")
    test_reg_preds, test_file_ids, _ = get_regression_predictions(regression_model, test_dataloader)
    test_class_probs, test_class_preds, _ = get_classifier_predictions(classifier_model, test_dataloader)
    
    # Always save individual model predictions 
    save_model_predictions(test_file_ids, test_reg_preds, f"{OUTPUT_DIR}/best_regressor_predictions.csv", "regressor")
    save_model_predictions(test_file_ids, test_class_preds, f"{OUTPUT_DIR}/best_classifier_predictions.csv", "classifier")
    
    # Save regressor predictions in format for submission
    regressor_submission_path = f"{OUTPUT_DIR}/regressor_submission.csv"
    rounded_preds = np.round(test_reg_preds).astype(int)
    century_preds = (rounded_preds // 10) + 1
    decade_preds = (rounded_preds % 10) + 1
    
    # Apply constraint: if century is 5, decade should not exceed 3
    for i in range(len(century_preds)):
        if century_preds[i] == 5 and decade_preds[i] > 3:
            decade_preds[i] = 3
            
    regressor_results = pd.DataFrame({
        'id': test_file_ids,
        'century_label': century_preds,
        'decade_label': decade_preds
    })
    regressor_results.to_csv(regressor_submission_path, index=False)
    print(f"Saved regressor submission to {regressor_submission_path}")
    
    # Save classifier predictions in format for submission
    classifier_submission_path = f"{OUTPUT_DIR}/classifier_submission.csv"
    century_preds = (test_class_preds // 10) + 1
    decade_preds = (test_class_preds % 10) + 1
    
    # Apply constraint: if century is 5, decade should not exceed 3
    for i in range(len(century_preds)):
        if century_preds[i] == 5 and decade_preds[i] > 3:
            decade_preds[i] = 3
            
    classifier_results = pd.DataFrame({
        'id': test_file_ids,
        'century_label': century_preds,
        'decade_label': decade_preds
    })
    classifier_results.to_csv(classifier_submission_path, index=False)
    print(f"Saved classifier submission to {classifier_submission_path}")
    
    # Compute prediction intervals using conformal prediction
    print(f"Computing prediction intervals with confidence level {args.confidence_level}...")
    lower_bounds, upper_bounds = get_prediction_intervals(test_reg_preds, nonconformity_scores, args.confidence_level)
    
    # Apply ensemble prediction to each sample
    print("Applying ensemble prediction...")
    ensemble_preds = [
        ensemble_prediction(reg_pred, lb, ub, class_probs) 
        for reg_pred, lb, ub, class_probs in zip(test_reg_preds, lower_bounds, upper_bounds, test_class_probs)
    ]
    ensemble_preds = np.array(ensemble_preds)
    
    # Convert absolute decade to century and decade
    print("Converting predictions to century and decade format...")
    century_preds = (ensemble_preds // 10) + 1
    decade_preds = (ensemble_preds % 10) + 1
    
    # Apply constraint: if century is 5, decade should not exceed 3
    for i in range(len(century_preds)):
        if century_preds[i] == 5 and decade_preds[i] > 3:
            decade_preds[i] = 3
    
    # Save ensemble predictions
    print(f"Saving ensemble predictions to {OUTPUT_DIR}/ensemble_predictions.csv...")
    results = pd.DataFrame({
        'id': test_file_ids,
        'century_label': century_preds,
        'decade_label': decade_preds,
        'absolute_prediction': ensemble_preds,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(f"{OUTPUT_DIR}/ensemble_predictions.csv"), exist_ok=True)
    results.to_csv(f"{OUTPUT_DIR}/ensemble_predictions.csv", index=False)
    
    # Also save in submission format
    ensemble_submission_path = f"{OUTPUT_DIR}/ensemble_submission.csv"
    submission = pd.DataFrame({
        'id': test_file_ids,
        'century_label': century_preds,
        'decade_label': decade_preds
    })
    submission.to_csv(ensemble_submission_path, index=False)
    print(f"Saved ensemble submission to {ensemble_submission_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()
