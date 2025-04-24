# Generate the tables for the report

import os
import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Update device setup for better multi-GPU support
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"Available GPUs: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Regressor model
class SingleHeadLongformerModel(nn.Module):
    def __init__(self):
        super(SingleHeadLongformerModel, self).__init__()
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
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
    
# Helper functions
def convert_to_absolute_decade(century, decade):
    """Convert century (0-4) and decade (0-9) to absolute decade (0-42)"""
    return century * 10 + decade

def mean_avg_error(y_true, y_pred):
    """Calculate Mean Absolute Error between true and predicted values"""
    return np.mean(np.abs(y_true - y_pred))

def evaluate_model(model, dataloader, name):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['decade_labels'].float().cpu().numpy())
    
    # Calculate metrics
    predictions_array = np.array(all_predictions)
    labels_array = np.array(all_labels)
    
    mae = mean_avg_error(labels_array, predictions_array)
    mse = np.mean((labels_array - predictions_array)**2)
    
    return {
        'name': name,
        'predictions': predictions_array,
        'mae': mae,
        'mse': mse
    }

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
        
        absolute_decade = convert_to_absolute_decade(century, decade)
        
        with open(os.path.join(self.path, file_path), 'r', encoding='utf-8') as f:
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
    

# Load data
texts_path = './data/Task2/texts'
#valid_path = os.path.join(texts_path, 'valid')

# Task 2.1
valid21 = pd.read_csv('./data/Task2/task2.1/valid.csv')

valid21.rename(columns={'label': 'century'}, inplace=True)
valid21['file_name'] = valid21['id']
valid21['id'] = valid21.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
valid21.set_index('id', inplace=True)

# Task 2.2
valid22 = pd.read_csv('./data/Task2/task2.2/valid.csv')

valid22.rename(columns={'label': 'century'}, inplace=True)
valid22['file_name'] = valid22['id']
valid22['id'] = valid22.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
valid22.set_index('id', inplace=True)

# Load blacklist
with open('blacklist.pkl', 'rb') as f:
    blacklist = pickle.load(f)

blacklist_valid = blacklist['valid']

X_valid_21 = [] # file names (.txt)
y_valid_21 = []
X_valid_22 = [] # file names (.txt)
y_valid_22 = []

# Process Task 2.1 data
for idx, row in valid21.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()

    if idx in blacklist_valid:
        continue

    # if 'gutenberg' in text.lower() and 'project' in text.lower():
    #     continue
        
    X_valid_21.append(file_name)
    y_valid_21.append(century-1)

# Process Task 2.2 data
for idx, row in valid22.iterrows():
    file_name = row.file_name
    century = row.century
    if idx in blacklist_valid:
        continue

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()

    # if 'gutenberg' in text.lower() and 'project' in text.lower():
    #     continue

    X_valid_22.append(file_name)
    y_valid_22.append(century-1)

print(f"Valid: {len(X_valid_21)}")

# Function to run evaluation with a specific random seed
def run_evaluation_with_seed(seed):
    print(f"\n=== Running evaluation with random seed {seed} ===")
    
    # Split the dataset before creating DataLoaders
    X_calib_paths, X_holdout_paths, y_calib_21, y_holdout_21, y_calib_22, y_holdout_22 = train_test_split(
        X_valid_21, y_valid_21, y_valid_22, test_size=0.4, random_state=seed
    )

    print(f"Calibration set: {len(X_calib_paths)} samples")
    print(f"Holdout set: {len(X_holdout_paths)} samples")

    calib_dataset = FileBasedSingleTaskDataset(X_calib_paths, texts_path, y_calib_21, y_calib_22, tokenizer)
    holdout_dataset = FileBasedSingleTaskDataset(X_holdout_paths, texts_path, y_holdout_21, y_holdout_22, tokenizer)

    calib_dataloader = DataLoader(calib_dataset, batch_size=16, shuffle=False)
    holdout_dataloader = DataLoader(holdout_dataset, batch_size=16, shuffle=False)
    
    model_maes = []
    
    for model_info in models:
        result = evaluate_model(model_info['model'], calib_dataloader, model_info['name'])
        model_maes.append(result['mae'])
        print(f"{model_info['name']}: MAE on calibration set = {result['mae']:.4f}")

    weights = 1 / np.array(model_maes)
    weights = weights / np.sum(weights)

    print(f"\nModel weights from calibration set (seed {seed}):")
    for i, (weight, model_info) in enumerate(zip(weights, models)):
        print(f"{model_info['name']}: {weight:.4f}")

    holdout_predictions = []
    holdout_labels_list = []

    with torch.no_grad():
        for batch in tqdm(holdout_dataloader, desc="Collecting holdout labels"):
            batch_labels = batch['decade_labels'].float().cpu().numpy()
            holdout_labels_list.extend(batch_labels)

    holdout_labels = np.array(holdout_labels_list)
    print(f"Collected {len(holdout_labels)} labels from holdout set")

    for model_info in models:
        model = model_info['model']
        model.eval()
        
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(holdout_dataloader, desc=f"Evaluating {model_info['name']} on holdout"):
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                all_preds.extend(predictions.cpu().numpy())
                
        holdout_predictions.append(np.array(all_preds))

    holdout_predictions = np.array(holdout_predictions)
    holdout_predictions = holdout_predictions.T

    epoch_4_idx = None
    for i, model_info in enumerate(models):
        if 'epoch_4' in model_info['name']:
            epoch_4_idx = i
            break

    # Get metrics for single regressor model
    regressor_preds = holdout_predictions[:, epoch_4_idx]
    regressor_mae = mean_avg_error(holdout_labels, regressor_preds)
    regressor_mse = np.mean((holdout_labels - regressor_preds)**2)

    regressor_preds_rounded = np.round(regressor_preds)
    regressor_preds_floored = np.floor(regressor_preds)
    regressor_mae_rounded = mean_avg_error(holdout_labels, regressor_preds_rounded)
    regressor_mse_rounded = np.mean((holdout_labels - regressor_preds_rounded)**2)
    regressor_mae_floored = mean_avg_error(holdout_labels, regressor_preds_floored)
    regressor_mse_floored = np.mean((holdout_labels - regressor_preds_floored)**2)

    # Ensemble predictions
    weighted_ensemble_predictions = np.zeros(len(holdout_labels))
    for i in range(len(models)):
        weighted_ensemble_predictions += weights[i] * holdout_predictions[:, i]

    # Simple average
    simple_ensemble_predictions = np.mean(holdout_predictions, axis=1)

    weighted_ensemble_predictions_rounded = np.round(weighted_ensemble_predictions)
    weighted_ensemble_predictions_floored = np.floor(weighted_ensemble_predictions)
    simple_ensemble_predictions_rounded = np.round(simple_ensemble_predictions)
    simple_ensemble_predictions_floored = np.floor(simple_ensemble_predictions)


    weighted_mae = mean_avg_error(holdout_labels, weighted_ensemble_predictions)
    weighted_mse = np.mean((holdout_labels - weighted_ensemble_predictions)**2)
    simple_mae = mean_avg_error(holdout_labels, simple_ensemble_predictions)
    simple_mse = np.mean((holdout_labels - simple_ensemble_predictions)**2)

    weighted_mae_rounded = mean_avg_error(holdout_labels, weighted_ensemble_predictions_rounded)
    weighted_mse_rounded = np.mean((holdout_labels - weighted_ensemble_predictions_rounded)**2)
    simple_mae_rounded = mean_avg_error(holdout_labels, simple_ensemble_predictions_rounded)
    simple_mse_rounded = np.mean((holdout_labels - simple_ensemble_predictions_rounded)**2)

    weighted_mae_floored = mean_avg_error(holdout_labels, weighted_ensemble_predictions_floored)
    weighted_mse_floored = np.mean((holdout_labels - weighted_ensemble_predictions_floored)**2)
    simple_mae_floored = mean_avg_error(holdout_labels, simple_ensemble_predictions_floored)
    simple_mse_floored = np.mean((holdout_labels - simple_ensemble_predictions_floored)**2)

    print(f"\nFinal Evaluation on Holdout Set (seed {seed}):")
    print("\nSingle Model (epoch_4):")
    print(f"  Raw: MAE = {regressor_mae:.4f}, MSE = {regressor_mse:.4f}")
    print(f"  Rounded: MAE = {regressor_mae_rounded:.4f}, MSE = {regressor_mse_rounded:.4f}")
    print(f"  Floored: MAE = {regressor_mae_floored:.4f}, MSE = {regressor_mse_floored:.4f}")

    print("\nEnsemble (average):")
    print(f"  Raw: MAE = {simple_mae:.4f}, MSE = {simple_mse:.4f}")
    print(f"  Rounded: MAE = {simple_mae_rounded:.4f}, MSE = {simple_mse_rounded:.4f}")
    print(f"  Floored: MAE = {simple_mae_floored:.4f}, MSE = {simple_mse_floored:.4f}")

    print("\nEnsemble (weighted):")
    print(f"  Raw: MAE = {weighted_mae:.4f}, MSE = {weighted_mse:.4f}")
    print(f"  Rounded: MAE = {weighted_mae_rounded:.4f}, MSE = {weighted_mse_rounded:.4f}")
    print(f"  Floored: MAE = {weighted_mae_floored:.4f}, MSE = {weighted_mse_floored:.4f}")
    
    # Return the metrics for this seed run
    return {
        'single_model': {
            'raw': {'mae': regressor_mae, 'mse': regressor_mse},
            'rounded': {'mae': regressor_mae_rounded, 'mse': regressor_mse_rounded},
            'floored': {'mae': regressor_mae_floored, 'mse': regressor_mse_floored}
        },
        'ensemble_avg': {
            'raw': {'mae': simple_mae, 'mse': simple_mse},
            'rounded': {'mae': simple_mae_rounded, 'mse': simple_mse_rounded},
            'floored': {'mae': simple_mae_floored, 'mse': simple_mse_floored}
        },
        'ensemble_weighted': {
            'raw': {'mae': weighted_mae, 'mse': weighted_mse},
            'rounded': {'mae': weighted_mae_rounded, 'mse': weighted_mse_rounded},
            'floored': {'mae': weighted_mae_floored, 'mse': weighted_mse_floored}
        }
    }

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
regressor = SingleHeadLongformerModel()
regressor.load_state_dict(torch.load('models/best_single_head_regression_model_epoch_4.pt'))

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    regressor = nn.DataParallel(regressor)

regressor.to(device)

# Find all model checkpoint files
model_dir = 'models/'
model_files = glob.glob(os.path.join(model_dir, 'best_single_head_regression_model_epoch_*.pt'))
print(f"Found {len(model_files)} model checkpoints:")
for file in model_files:
    print(f"  - {os.path.basename(file)}")

# Load models
models = []

for model_path in model_files:
    model = SingleHeadLongformerModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel for {os.path.basename(model_path)}")
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    models.append({
        'name': os.path.basename(model_path),
        'model': model
    })
    print(f"Loaded {os.path.basename(model_path)}")

# Run evaluation with multiple seeds
random_seeds = [0, 42, 363, 1111, 1408]
all_results = {}

for seed in random_seeds:
    all_results[seed] = run_evaluation_with_seed(seed)

print("\n\n============= EVALUATING =============")
print(f"Results across {len(random_seeds)} random seeds: {random_seeds}")

metrics = ['mae', 'mse']
model_types = ['single_model', 'ensemble_avg', 'ensemble_weighted']
prediction_types = ['raw', 'rounded', 'floored']

stats_rows = []

for model_type in model_types:
    for pred_type in prediction_types:
        for metric in metrics:
            values = [all_results[seed][model_type][pred_type][metric] for seed in random_seeds]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            stats_rows.append({
                'Model': model_type,
                'Prediction': pred_type,
                'Metric': metric,
                'Mean': mean_val,
                'Std': std_val,
                'Min': min_val,
                'Max': max_val,
                'Range': max_val - min_val
            })
            
            print(f"{model_type} - {pred_type} - {metric}:")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Std Dev: {std_val:.4f}")
            print(f"  Min: {min_val:.4f}")
            print(f"  Max: {max_val:.4f}")
            print(f"  Range: {max_val - min_val:.4f}")
            print()

stats_df = pd.DataFrame(stats_rows)
print("\nStatistics Summary:")
print(stats_df)

stats_df.to_csv('seed_variance_analysis.csv', index=False)
print("Statistics saved to 'seed_variance_analysis.csv'")

plt.figure(figsize=(14, 10))
sns.boxplot(x="Model", y="Mean", hue="Prediction", data=stats_df[stats_df['Metric'] == 'mae'])
plt.title('MAE Distribution Across Different Seeds')
plt.savefig('mae_seed_variance.png')

plt.figure(figsize=(14, 10))
sns.boxplot(x="Model", y="Mean", hue="Prediction", data=stats_df[stats_df['Metric'] == 'mse'])
plt.title('MSE Distribution Across Different Seeds')
plt.savefig('mse_seed_variance.png')
print("Visualizations saved to 'mae_seed_variance.png' and 'mse_seed_variance.png'")
