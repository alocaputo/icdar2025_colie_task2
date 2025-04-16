import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import pickle
import random

"""
python task2x_regression_ensemble.py --mode prepare --k 5

python task2x_regression_ensemble.py --mode train --fold 0 --gpu 0

python task2x_regression_ensemble.py --mode evaluate --gpu 0

"""
# Set seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['NUMPY_SEED'] = str(SEED)
np.random.default_rng(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load data
texts_path = './data/Task2/texts'

train_path = os.path.join(texts_path, 'train')
valid_path = os.path.join(texts_path, 'valid')

train_files = os.listdir(train_path)
valid_files = os.listdir(valid_path)

print(f"Train: {len(train_files)}, Test: {len(valid_files)}")

# Task 2.1
train21 = pd.read_csv('./data/Task2/task2.1/train.csv')
valid21 = pd.read_csv('./data/Task2/task2.1/valid.csv')

train21.rename(columns={'label': 'century'}, inplace=True)
train21['file_name'] = train21['id']
train21['id'] = train21.id.str.replace('train_text', '').str.replace('.txt', '').astype(int)
train21.set_index('id', inplace=True)

valid21.rename(columns={'label': 'century'}, inplace=True)
valid21['file_name'] = valid21['id']
valid21['id'] = valid21.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
valid21.set_index('id', inplace=True)

# Task 2.2
train22 = pd.read_csv('./data/Task2/task2.2/train.csv')
valid22 = pd.read_csv('./data/Task2/task2.2/valid.csv')

train22.rename(columns={'label': 'century'}, inplace=True)
train22['file_name'] = train22['id']
train22['id'] = train22.id.str.replace('train_text', '').str.replace('.txt', '').astype(int)
train22.set_index('id', inplace=True)

valid22.rename(columns={'label': 'century'}, inplace=True)
valid22['file_name'] = valid22['id']
valid22['id'] = valid22.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
valid22.set_index('id', inplace=True)

with open('blacklist.pkl', 'rb') as f:
    blacklist = pickle.load(f)

blacklist_train = blacklist['train']
blacklist_valid = blacklist['valid']

X_train_21 = [] # file names (.txt)
y_train_21 = [] 

X_valid_21 = [] # file names (.txt)
y_valid_21 = []

for idx, row in train21.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(train_path, file_name), 'r') as file:
        text = file.read()
    if 'gutenberg' in text.lower():
        continue

    if idx in blacklist_train:
        continue

    X_train_21.append(file_name)
    y_train_21.append(century-1)

for idx, row in valid21.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(valid_path, file_name), 'r') as file:
        text = file.read()
    if 'gutenberg' in text.lower():
        continue

    if idx in blacklist_valid:
        continue
        
    X_valid_21.append(file_name)
    y_valid_21.append(century-1)

X_train_22 = [] # file names (.txt)
y_train_22 = []

X_valid_22 = [] # file names (.txt)
y_valid_22 = []

for idx, row in train22.iterrows():
    file_name = row.file_name
    century = row.century
    if idx in blacklist_train:
        continue

    with open(os.path.join(train_path, file_name), 'r') as file:
        text = file.read()
    if 'gutenberg' in text.lower():
        continue

    if idx in blacklist_train:
        continue

    X_train_22.append(file_name)
    y_train_22.append(century-1)

for idx, row in valid22.iterrows():
    file_name = row.file_name
    century = row.century
    if idx in blacklist_valid:
        continue

    with open(os.path.join(valid_path, file_name), 'r') as file:
        text = file.read()
    if 'gutenberg' in text.lower():
        continue
    
    if idx in blacklist_valid:
        continue

    X_valid_22.append(file_name)
    y_valid_22.append(century-1)

# Double check the order of the data
for x21, x22 in zip(X_train_21, X_train_22):
    assert x21 == x22

for x21, x22 in zip(X_valid_21, X_valid_22):
    assert x21 == x22

print('Data loaded successfully!')
print(f"Train: {len(X_train_21)}, Test: {len(X_valid_21)}")

def convert_to_absolute_decade(century, decade):
    """Convert century (0-4) and decade (0-9) to absolute decade (0-42)"""
    return century * 10 + decade

class FileBasedSingleTaskDataset(Dataset):
    def __init__(self, file_paths, path, century_labels, decade_labels, tokenizer, max_length=1536):
        self.file_paths = file_paths
        self.path = path
        self.century_labels = century_labels
        self.decade_labels = decade_labels
        self.tokenizer = tokenizer
        self.max_length = max_length            # default 512*3 = 1536
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        century = self.century_labels[idx]
        decade = self.decade_labels[idx]
        
        # Convert to absolute decade label
        absolute_decade = convert_to_absolute_decade(century, decade)
        
        # Handle both string paths and path functions
        if callable(self.path):
            # If path is a function, call it with the file path
            full_path = self.path(file_path)
        else:
            # If path is a string, use os.path.join
            full_path = os.path.join(self.path, file_path)
            
        with open(full_path, 'r', encoding='utf-8') as f:
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

def mean_avg_error(y_true, y_pred):
    """Calculate Mean Absolute Error between true and predicted values"""
    return np.mean(np.abs(y_true - y_pred))

def evaluate_single_head(model, dataloader, device):
    """Evaluate model performance
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with validation data
        device: Device to run evaluation on
    """
    model.eval()
    total_predictions = 0
    preds = []
    labels_list = []
    total_loss = 0
    batch_count = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            decade_predictions = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            # Convert labels to float for regression
            labels = batch['decade_labels'].float()
            
            # Calculate loss
            loss = criterion(decade_predictions, labels)
            total_loss += loss.item()
            batch_count += 1
            
            total_predictions += len(decade_predictions)
            preds.extend(decade_predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    # Calculate metrics
    average_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    mae = mean_avg_error(np.array(labels_list), np.array(preds))
    mse = np.mean((np.array(labels_list) - np.array(preds))**2)
    
    return average_loss, mae, mse

def train_model_on_fold(model, train_loader, val_loader, fold_num, epochs=10, device=None, resume=False):
    """Train a single model on a specific fold with resumable training support"""
    if device is not None:
        model = model.to(device)
    
    model_save_dir = f'models/task2x_ensemble/fold_{fold_num}'
    os.makedirs(model_save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_save_dir, 'checkpoint.pt')
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_tolerance = 3
    epochs_no_improve = 0
    
    # Check if checkpoint exists and resume is enabled
    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint for fold {fold_num}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        if 'best_model_state' in checkpoint:
            best_model_state = checkpoint['best_model_state']
        
        print(f"Resuming training from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Fold {fold_num}, Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            decade_predictions = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            labels = batch['decade_labels'].float()
            loss = criterion(decade_predictions, labels)
            loss.backward()
            optimizer.step()
        
        val_loss, val_mae, val_mse = evaluate_single_head(model, val_loader, device)
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve,
            'val_metrics': {
                'loss': val_loss,
                'mae': val_mae,
                'mse': val_mse
            }
        }
        
        if best_model_state is not None:
            checkpoint['best_model_state'] = best_model_state
            
        torch.save(checkpoint, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            model_path = os.path.join(model_save_dir, f'best_model_epoch_{epoch+1}.pt')
            torch.save(best_model_state, model_path)
            
            # Update checkpoint with best model state
            checkpoint['best_model_state'] = best_model_state
            checkpoint['best_val_loss'] = best_val_loss
            checkpoint['epochs_no_improve'] = 0
            torch.save(checkpoint, checkpoint_path)
            
            print(f"Fold {fold_num}, Epoch {epoch+1}: New best model saved with validation loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Fold {fold_num}, Epoch {epoch+1}: No improvement for {epochs_no_improve} epochs.")
            
            if epochs_no_improve >= early_stopping_tolerance:
                print(f"Fold {fold_num}: Early stopping triggered after {epoch+1} epochs.")
                break
        
        print(f"Fold {fold_num}, Epoch {epoch+1}: MAE = {val_mae:.4f}, MSE = {val_mse:.4f}, Val Loss = {val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        final_model_path = os.path.join(model_save_dir, 'best_model.pt')
        torch.save(best_model_state, final_model_path)
    
    # Clean up checkpoint if training completes successfully
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return model, final_model_path

def get_device(gpu_id):
    """Get appropriate device, with fallback to CPU for invalid GPU IDs
    
    Args:
        gpu_id: Requested GPU ID (-1 for CPU)
    
    Returns:
        torch.device: The selected device
    """
    if not torch.cuda.is_available() or gpu_id < 0:
        print(f"CUDA not available or CPU requested. Using CPU.")
        return torch.device("cpu")
    
    # Check if the requested GPU ID is valid
    num_gpus = torch.cuda.device_count()
    if gpu_id >= num_gpus:
        print(f"Warning: GPU {gpu_id} not available, only {num_gpus} GPUs found.")
        print(f"Falling back to GPU 0")
        gpu_id = 0  # Fall back to GPU 0
        if num_gpus == 0:
            print(f"No GPUs available. Using CPU.")
            return torch.device("cpu")
    
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using device: {device} ({torch.cuda.get_device_name(gpu_id)})")
    return device

def train_specific_fold(fold_num, gpu_id, X_files, file_path_func, century_labels, decade_labels, tokenizer, k=5, epochs=10, resume=True):
    """Train a specific fold on a specific GPU with resumable capability
    
    Args:
        file_path_func: Function that takes a file name and returns full path
        resume: Whether to resume training from checkpoint if available
    """
    # Load fold indices
    try:
        with open(f'folds/fold_indices_{k}.pkl', 'rb') as f:
            fold_indices = pickle.load(f)
    except FileNotFoundError:
        print(f"Fold indices file not found. Creating new fold indices for k={k}...")
        fold_indices = create_fold_indices(X_files, k)
    
    if fold_num < 0 or fold_num >= k:
        raise ValueError(f"Invalid fold number. Must be between 0 and {k-1}")
        
    # Use the get_device function to safely get a device
    device = get_device(gpu_id)
    print(f"Training fold {fold_num} on {device}")
    
    # Get train and validation indices for this fold
    train_idx, val_idx = fold_indices[fold_num]
    
    # Create dataset for this fold - pass file_path_func directly
    X_train_fold = [X_files[i] for i in train_idx]
    y_train_century_fold = [century_labels[i] for i in train_idx]
    y_train_decade_fold = [decade_labels[i] for i in train_idx]
    
    X_val_fold = [X_files[i] for i in val_idx]
    y_val_century_fold = [century_labels[i] for i in val_idx]
    y_val_decade_fold = [decade_labels[i] for i in val_idx]
    
    train_fold_dataset = FileBasedSingleTaskDataset(
        X_train_fold, file_path_func, y_train_century_fold, y_train_decade_fold, tokenizer
    )
    val_fold_dataset = FileBasedSingleTaskDataset(
        X_val_fold, file_path_func, y_val_century_fold, y_val_decade_fold, tokenizer
    )
    
    batch_size = 16
    train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize and train model
    fold_model = SingleHeadLongformerModel()
    fold_model = fold_model.to(device)
    
    _, model_path = train_model_on_fold(
        fold_model, train_loader, val_loader, fold_num+1, epochs=epochs, device=device, resume=resume
    )
    
    return model_path

def create_fold_indices(X_files, k=5):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(X_files))
    with open(f'folds/fold_indices_{k}.pkl', 'wb') as f:
        pickle.dump(fold_indices, f)
    return fold_indices

# Update create_k_fold_ensemble to use function properly
def create_k_fold_ensemble(X_files, file_path_func, century_labels, decade_labels, tokenizer, k=5, epochs=10, gpu_ids=None):
    """Create an ensemble of models using k-fold cross-validation
    
    Args:
        file_path_func: Function that takes a file name and returns full path
        gpu_ids: List of GPU IDs to use for distributed training. If None, use current device.
    """
    from sklearn.model_selection import KFold
    
    # Combine all data
    all_files = X_files
    all_century_labels = century_labels
    all_decade_labels = decade_labels
    
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(all_files))
    
    ensemble_models = []
    ensemble_paths = []
    
    # Check if we're using distributed training
    using_distributed = gpu_ids is not None and len(gpu_ids) > 0
    
    for fold_num, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n=== Training Fold {fold_num+1}/{k} ===")
        
        # Create dataset for this fold
        X_train_fold = [all_files[i] for i in train_idx]
        y_train_century_fold = [all_century_labels[i] for i in train_idx]
        y_train_decade_fold = [all_decade_labels[i] for i in train_idx]
        
        X_val_fold = [all_files[i] for i in val_idx]
        y_val_century_fold = [all_century_labels[i] for i in val_idx]
        y_val_decade_fold = [all_decade_labels[i] for i in val_idx]
        
        train_fold_dataset = FileBasedSingleTaskDataset(
            X_train_fold, file_path_func, y_train_century_fold, y_train_decade_fold, tokenizer
        )
        val_fold_dataset = FileBasedSingleTaskDataset(
            X_val_fold, file_path_func, y_val_century_fold, y_val_decade_fold, tokenizer
        )
        
        batch_size = 16
        train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)
        
        # Set device for this fold with validation
        fold_device = None
        if using_distributed:
            gpu_idx = fold_num % len(gpu_ids)  # Distribute folds across available GPUs
            fold_device = get_device(gpu_ids[gpu_idx])
            print(f"Training fold {fold_num+1} on {fold_device}")
        
        # Initialize and train a new model for this fold
        fold_model = SingleHeadLongformerModel()
        if fold_device:
            fold_model = fold_model.to(fold_device)
        else:
            fold_model = fold_model.to(device)
        
        trained_model, model_path = train_model_on_fold(
            fold_model, train_loader, val_loader, fold_num+1, epochs=epochs, device=fold_device
        )
        
        ensemble_models.append(trained_model)
        ensemble_paths.append(model_path)
        
        # Free up memory
        del fold_model, trained_model
        torch.cuda.empty_cache()
    
    return ensemble_paths

def ensemble_predict(model_paths, dataloader, device):
    """Make predictions using an ensemble of models"""
    all_predictions = []
    
    for path in model_paths:
        # Load the model
        model = SingleHeadLongformerModel()
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        
        batch_predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Predicting with model from {path}"):
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
    return ensemble_predictions

def evaluate_ensemble(model_paths, dataloader, device):
    """Evaluate the ensemble on a dataset"""
    # Get true labels
    all_labels = []
    for batch in dataloader:
        labels = batch['decade_labels'].numpy()
        all_labels.extend(labels)
    
    # Get predictions
    predictions = ensemble_predict(model_paths, dataloader, device)
    
    # Calculate metrics
    mae = mean_avg_error(np.array(all_labels), predictions)
    mse = np.mean((np.array(all_labels) - predictions)**2)
    
    return mae, mse

def get_all_trained_models(k=5):
    """Get all trained models for the ensemble"""
    model_paths = []
    for fold in range(k):
        fold_model_path = f'models/task2x_ensemble/fold_{fold+1}/best_model.pt'
        if os.path.exists(fold_model_path):
            model_paths.append(fold_model_path)
        else:
            print(f"Warning: Model for fold {fold+1} not found at {fold_model_path}")
    
    return model_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate ensemble models for ICDAR 2025 Task 2X')
    parser.add_argument('--mode', type=str, choices=['prepare', 'train', 'evaluate', 'full'], default='full',
                        help='Mode: prepare folds, train specific fold, evaluate ensemble, or full pipeline')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to train (0-indexed)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--k', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint if available')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Start training from scratch')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    # Combine train and validation data
    all_files = X_train_21 + X_valid_21
    all_century_labels = y_train_21 + y_valid_21
    all_decade_labels = y_train_22 + y_valid_22
    
    # Define file path function
    combined_path = {**{file: os.path.join(train_path, file) for file in X_train_21}, 
                    **{file: os.path.join(valid_path, file) for file in X_valid_21}}
    
    # Fix for file_path_func
    def get_file_path(file):
        return combined_path.get(file, os.path.join(train_path, file))
    
    file_path_func = get_file_path
    
    # Create a separate validation set for final evaluation
    X_final_train, X_final_valid, y_final_train_century, y_final_valid_century, y_final_train_decade, y_final_valid_decade = train_test_split(
        all_files, all_century_labels, all_decade_labels, test_size=0.1, random_state=SEED
    )
    
    # Display available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA devices:")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Will use CPU.")
    
    if args.mode == 'prepare' or args.mode == 'full':
        print(f"Creating {args.k}-fold cross-validation indices...")
        create_fold_indices(X_final_train, k=args.k)
    
    if args.mode == 'train' or (args.mode == 'full' and args.fold >= 0):
        print(f"Training fold {args.fold} on GPU {args.gpu}...")
        model_path = train_specific_fold(
            args.fold, 
            args.gpu, 
            X_final_train, 
            file_path_func, 
            y_final_train_century, 
            y_final_train_decade,
            tokenizer, 
            k=args.k,
            epochs=args.epochs,
            resume=args.resume
        )
        print(f"Finished training fold {args.fold}. Model saved at {model_path}")
    
    if args.mode == 'evaluate' or args.mode == 'full':
        # Set device for evaluation using the get_device function
        device = get_device(args.gpu)
        
        # Get all trained models
        ensemble_paths = get_all_trained_models(args.k)
        
        if not ensemble_paths:
            print("No trained models found. Cannot evaluate ensemble.")
        else:
            print(f"Found {len(ensemble_paths)} trained models for ensemble evaluation.")
            
            # Create final validation dataset
            final_valid_dataset = FileBasedSingleTaskDataset(
                X_final_valid,
                file_path_func,
                y_final_valid_century,
                y_final_valid_decade,
                tokenizer
            )
            
            final_valid_dataloader = DataLoader(final_valid_dataset, batch_size=16, shuffle=False)
            
            # Evaluate the ensemble
            print("Evaluating ensemble performance...")
            final_mae, final_mse = evaluate_ensemble(ensemble_paths, final_valid_dataloader, device)
            print(f"Ensemble Final Results: MAE = {final_mae:.4f}, MSE = {final_mse:.4f}")
    
    print("Process complete.")