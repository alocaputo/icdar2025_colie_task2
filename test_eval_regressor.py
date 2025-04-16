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
from sklearn.isotonic import IsotonicRegression

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ConformalCalibrator:
    """Calibrates model predictions using conformal prediction techniques"""
    
    def __init__(self):
        self.calibration_model = IsotonicRegression(out_of_bounds='clip')
        self.error_quantiles = {}
        self.calibrated = False
    
    def calibrate(self, true_values, predicted_values, quantiles=[0.9]):
        """Calibrate the model using a calibration set
        
        Args:
            true_values: Ground truth labels
            predicted_values: Model's raw predictions
            quantiles: List of quantiles to compute
        """
        # Convert to numpy arrays if needed
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)
        
        # Train isotonic regression to calibrate predictions
        self.calibration_model.fit(predicted_values, true_values)
        
        # Calculate residuals after calibration
        calibrated_preds = self.calibration_model.predict(predicted_values)
        residuals = true_values - calibrated_preds
        
        # Store quantiles of the absolute errors for different uncertainty levels
        for q in quantiles:
            self.error_quantiles[q] = np.quantile(np.abs(residuals), q)
        
        self.calibrated = True
        return self
    
    def predict(self, raw_predictions):
        """Return calibrated predictions
        
        Args:
            raw_predictions: Model's raw predictions
            
        Returns:
            Calibrated predictions
        """
        if not self.calibrated:
            return raw_predictions
            
        return self.calibration_model.predict(raw_predictions)
    
    def predict_with_intervals(self, raw_predictions, confidence=0.9):
        """Return calibrated predictions with intervals
        
        Args:
            raw_predictions: Model's raw predictions
            confidence: Confidence level (default: 0.9)
            
        Returns:
            Tuple of (calibrated predictions, lower bounds, upper bounds)
        """
        if not self.calibrated:
            return raw_predictions, None, None
            
        calibrated = self.predict(raw_predictions)
        q = self.error_quantiles.get(confidence, 
             # If quantile not pre-computed, use the closest available
             self.error_quantiles[min(self.error_quantiles.keys(), 
                                     key=lambda x: abs(x-confidence))])
        
        lower = calibrated - q
        upper = calibrated + q
        
        return calibrated, lower, upper

def convert_absolute_decade_to_century_decade(absolute_decade):
    """Convert absolute decade (0-43) to century (1-5) and decade (1-10)"""
    # Ensure the value is an integer in the valid range
    absolute_decade = max(0, min(43, int(round(absolute_decade))))
    
    # Convert to century (1-indexed) and decade (1-10)
    century = (absolute_decade // 10) + 1
    decade = (absolute_decade % 10) + 1  # Convert from 0-9 to 1-10
    
    return century, decade

def run_inference(model, dataloader, device):
    """Run standard inference on the model"""
    model.eval()
    file_ids = []
    century_preds = []
    decade_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Model outputs a single value (absolute decade)
            absolute_decade = model(input_ids, attention_mask)
            
            # Convert to numpy array
            if isinstance(absolute_decade, torch.Tensor):
                absolute_decade = absolute_decade.cpu().numpy()
            
            # Process each prediction in the batch
            for abs_dec in absolute_decade:
                century, decade = convert_absolute_decade_to_century_decade(abs_dec)
                century_preds.append(century)
                decade_preds.append(decade)
            
            file_ids.extend(batch['file_id'])
    
    return file_ids, century_preds, decade_preds

def compute_conformal_scores_regression(model, dataloader, device):
    """Compute conformal scores for regression model"""
    model.eval()
    true_values = []
    pred_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing conformal scores"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Model outputs a single value (absolute decade)
            predictions = model(input_ids, attention_mask)
            
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            # Extract ground truth labels
            true_century = batch['century_label'].cpu().numpy()
            
            # For validation data, we have the century (0-indexed)
            # Convert true century (0-indexed) to absolute decade range for comparison
            # We use the middle decade (5) of each century as an approximation
            true_absolute_decade = true_century * 10 + 5
            
            true_values.extend(true_absolute_decade.tolist())
            
            if isinstance(predictions, np.ndarray):
                pred_values.extend(predictions.tolist())
            else:
                # Handle scalar output
                pred_values.append(float(predictions))
    
    return np.array(true_values), np.array(pred_values)

def generate_conformal_results(model, dataloader, device, calibrator, confidence=0.9):
    """Generate results using conformal prediction for regression"""
    model.eval()
    file_ids = []
    century_preds = []
    decade_preds = []
    prediction_intervals = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating conformal predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Model outputs a single value (absolute decade)
            predictions = model(input_ids, attention_mask)
            
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            # Apply conformal calibration
            calibrated, lower, upper = calibrator.predict_with_intervals(predictions, confidence)
            
            # Process each prediction
            for cal, lo, hi in zip(calibrated, lower, upper):
                # Convert calibrated prediction to century and decade
                century, decade = convert_absolute_decade_to_century_decade(cal)
                
                # Convert intervals to century
                lo_century = max(1, min(5, int((lo // 10) + 1)))
                hi_century = max(1, min(5, int((hi // 10) + 1)))
                
                century_preds.append(century)
                decade_preds.append(decade)
                prediction_intervals.append((lo_century, hi_century))
            
            file_ids.extend(batch['file_id'])
    
    # Create results DataFrame
    results = pd.DataFrame({
        'id': file_ids,
        'century_label': century_preds,
        'decade_label': decade_preds
    })
    
    # Add prediction intervals
    results['century_lower'] = [interval[0] for interval in prediction_intervals]
    results['century_upper'] = [interval[1] for interval in prediction_intervals]
    
    # Apply constraints
    results.loc[results['century_label'] == 5, 'decade_label'] = results.loc[results['century_label'] == 5, 'decade_label'].clip(upper=3)
    
    # Ensure integer types for all numerical columns
    results['century_label'] = results['century_label'].astype(int)
    results['decade_label'] = results['decade_label'].astype(int)
    results['century_lower'] = results['century_lower'].astype(int)
    results['century_upper'] = results['century_upper'].astype(int)
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test evaluation script for ICDAR 2025 Task 2')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for conformal prediction (1-confidence)')
#parser.add_argument('--rounding', type=str, default='round', choices=['round', 'floor', 'ceil'], 
    #                   help='Rounding method for regression outputs')
    args = parser.parse_args()
    
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
    print(f"Evaluating regression model")
    model_path = args.model_path

    from task2x_combined_regression import SingleHeadLongformerModel
    model = SingleHeadLongformerModel()
    output_file_base = './submissions/regression'
    print("Loaded regression model architecture")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded regression model from {model_path}")
    
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    
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
    
    # Initialize conformal calibrator
    print("Computing regression conformal scores...")
    calibrator = ConformalCalibrator()
    
    # Get true values and predictions for calibration
    true_values, pred_values = compute_conformal_scores_regression(model, valid_dataloader, device)
    
    # Calibrate the model
    confidence = 1 - args.alpha
    calibrator.calibrate(true_values, pred_values, quantiles=[confidence])
    print(f"Model calibrated, error quantile at {confidence} confidence: {calibrator.error_quantiles[confidence]:.4f}")
    
    # Generate conformal predictions
    results = generate_conformal_results(model, test_dataloader, device, calibrator, confidence)
    
    # Create output directory if it doesn't exist
    output_dir = './submissions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model filename for the output CSV name
    model_filename = os.path.basename(args.model_path).replace('.pt', '')
    
    # Fix the path construction
    output_file = os.path.join(output_dir, f"cp_regression_{model_filename}_conformal_alpha{args.alpha}.csv")
    
    # Save results - drop the extra columns for submission
    submission_results = results[['id', 'century_label', 'decade_label']]
    submission_results.to_csv(output_file, index=False)
    
    # Also save the full results with confidence intervals
    full_output_file = os.path.join(output_dir, f"cp_regression_{model_filename}_conformal_alpha{args.alpha}_full.csv")
    results.to_csv(full_output_file, index=False)
    
    print(f"Saved submission predictions to {output_file}")
    print(f"Saved full predictions with intervals to {full_output_file}")