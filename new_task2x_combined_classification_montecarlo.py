import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import pickle
import random
import argparse
import json
from datetime import datetime

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a specific model using Monte Carlo CV')
    
    # Only keep the model ID parameter
    parser.add_argument('--model_id', type=int, required=True,
                        help='ID of the model to train (0-N)')
    
    return parser.parse_args()

# Load data
texts_path = './data/Task2/texts'

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

print(f"Train: {len(train21)}, Test: {len(valid21)}")

X_train_21 = [] # file names (.txt)
y_train_21 = [] 

X_valid_21 = [] # file names (.txt)
y_valid_21 = []

for idx, row in train21.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()
    # if 'project' in text.lower() and 'gutenberg' in text.lower():
    #     continue

    X_train_21.append(file_name)
    y_train_21.append(century-1)

for idx, row in valid21.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()
    # if 'project' in text.lower() and 'gutenberg' in text.lower():
    #     continue
        
    X_valid_21.append(file_name)
    y_valid_21.append(century-1)

X_train_22 = [] # file names (.txt)
y_train_22 = []

X_valid_22 = [] # file names (.txt)
y_valid_22 = []

for idx, row in train22.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()
    # if 'project' in text.lower() and 'gutenberg' in text.lower():
    #     continue

    X_train_22.append(file_name)
    y_train_22.append(century-1)

for idx, row in valid22.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()
    # if 'project' in text.lower() and 'gutenberg' in text.lower():
    #     continue
    
    X_valid_22.append(file_name)
    y_valid_22.append(century-1)

for x21, x22 in zip(X_train_21, X_train_22):
    assert x21 == x22

for x21, x22 in zip(X_valid_21, X_valid_22):
    assert x21 == x22

print('Data loaded successfully!')
print(f"Train: {len(X_train_21)}, Test: {len(X_valid_21)}")

def convert_to_absolute_decade(century, decade):
    """Convert century (0-4) and decade (0-9) to absolute decade (0-42)"""
    return century * 10 + decade

class MCFileBasedDataset(Dataset):
    def __init__(self, file_paths, path, century_labels, decade_labels, tokenizer, max_length=1536):
        self.file_paths = file_paths
        self.path = path
        self.century_labels = century_labels
        self.decade_labels = decade_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-load all texts to avoid reopening files
        self.texts = []
        print("Loading texts...")
        for file_path in tqdm(file_paths):
            with open(os.path.join(path, file_path), 'r', encoding='utf-8', errors='replace') as f:
                self.texts.append(f.read())
                
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        century = self.century_labels[idx]
        decade = self.decade_labels[idx]
        
        # Convert to absolute decade label
        absolute_decade = convert_to_absolute_decade(century, decade)
        
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

class OrdinalClassificationLoss(nn.Module):
    def __init__(self, num_classes=43, q=1.0, entropy_weight=0.1, loss_weight=1.0, 
                 annealing=True, annealing_epochs=10, initial_temperature=1.0):
        super(OrdinalClassificationLoss, self).__init__()
        self.num_classes = num_classes
        self.q = q  # Power for distance penalty (q=1 is MAE, q=2 is MSE)
        self.entropy_weight = entropy_weight  # Weight for entropy regularization
        self.loss_weight = loss_weight  # Weight for distance penalty
        self.annealing = annealing  # Whether to use annealing for entropy weight
        self.annealing_epochs = annealing_epochs  # Number of epochs for annealing
        self.initial_temperature = initial_temperature  # Initial temperature for annealing
        self.current_epoch = 0  # Current epoch for annealing
        self.smax = nn.Softmax(dim=1)
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, logits, targets):
        device = logits.device
        probs = self.smax(logits)
        log_vals = torch.log(probs + 1e-10)
        neg_entropy = torch.sum(torch.sum(probs * log_vals, dim=1))
        
        if self.annealing:
            current_entropy_weight = max(0, (self.initial_temperature * 
                                            (1 - self.current_epoch / self.annealing_epochs)))
        else:
            current_entropy_weight = self.entropy_weight
        
        batch_size = targets.size(0)
        range_vals = torch.arange(0, self.num_classes, dtype=torch.float, device=device)
        targets = targets.to(device)
        expanded_targets = targets.view(-1, 1).expand(batch_size, self.num_classes).float()
        expanded_range = range_vals.view(1, -1).expand(batch_size, -1)
        
        distance_penalty = torch.pow(torch.abs(expanded_range - expanded_targets), self.q)
        weighted_penalty = torch.sum(probs * distance_penalty)
        
        loss = (neg_entropy * current_entropy_weight) + (self.loss_weight * weighted_penalty)
        
        return loss

class SingleHeadLongformerModel(nn.Module):
    def __init__(self):
        super(SingleHeadLongformerModel, self).__init__()
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        self.decade_classifier = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 43)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        decade_logits = self.decade_classifier(pooled_output)
        
        return decade_logits

def mean_avg_error(y_true, y_pred):
    """Calculate Mean Absolute Error between true and predicted values"""
    return np.mean(np.abs(y_true - y_pred))

def evaluate_single_head(model, dataloader, criterion, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    total_predictions = 0
    preds = []
    labels_list = []
    total_loss = 0
    batch_count = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            labels = batch['decade_labels']
            
            loss = criterion(decade_logits, labels)
            total_loss += loss.item()
            batch_count += 1
            
            probs = nn.Softmax(dim=1)(decade_logits)
            _, predicted_classes = torch.max(probs, dim=1)
            
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += len(labels)
            
            preds.extend(predicted_classes.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    average_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    mae = mean_avg_error(np.array(labels_list), np.array(preds))
    
    return average_loss, accuracy, mae

def train_single_model(model, train_loader, val_loader, criterion, optimizer, save_dir, 
                      epochs=10, early_stop_patience=2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_val_mae = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    best_metrics = None
    
    for epoch in range(epochs):
        criterion.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            loss = criterion(decade_logits, batch['decade_labels'])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        val_loss, val_accuracy, val_mae = evaluate_single_head(model, val_loader, criterion, device)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pt'))
            
            best_metrics = {
                "loss": val_loss,
                "accuracy": val_accuracy,
                "mae": val_mae,
                "epoch": epoch + 1
            }
            
            print(f"Epoch {epoch+1}: New best model saved with val MAE: {val_mae:.4f}, loss: {val_loss:.4f}, acc: {val_accuracy:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: Val MAE: {val_mae:.4f}, loss: {val_loss:.4f}, acc: {val_accuracy:.4f} (no improvement for {epochs_no_improve} epochs)")
            
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=4)
    
    return model, best_metrics

def train_model_with_id(model_id, full_dataset):
    """Train a single model with the specified ID"""
    
    # Fixed parameters
    batch_size = 16
    epochs = 10
    learning_rate = 2e-5
    early_stop_patience = 2
    max_length = 1536
    validation_size = 0.2
    entropy_weight = 0.1
    loss_weight = 1.0
    q_value = 1.0
    output_dir = 'new_models/montecarlo_classifiers'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save fixed configuration parameters for reference
    config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'early_stop_patience': early_stop_patience,
        'max_length': max_length,
        'validation_size': validation_size,
        'entropy_weight': entropy_weight, 
        'loss_weight': loss_weight,
        'q_value': q_value
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    print(f"\n===== Training Model ID: {model_id} =====")
    
    # Set seeds specifically for this model ID to ensure different splits
    local_seed = SEED + model_id
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)
    
    # Model filename with ID
    model_filename = f"model_{model_id}"
    
    # Create directory for this model if needed
    model_dir = os.path.join(output_dir, model_filename)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create the dataset split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    split_idx = int(np.floor(validation_size * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        sampler=valid_sampler
    )
    
    # Initialize model
    model = SingleHeadLongformerModel()
    model.to(device)
    
    # Initialize criterion
    criterion = OrdinalClassificationLoss(
        num_classes=43, 
        q=q_value,
        entropy_weight=entropy_weight,
        loss_weight=loss_weight,
        annealing=True,
        annealing_epochs=epochs,
        initial_temperature=1.0
    )
    criterion = criterion.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train model
    model, val_metrics = train_single_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        save_dir=model_dir,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        device=device
    )
    
    # Save split indices for reproducibility
    split_info = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "seed": local_seed
    }
    
    with open(os.path.join(model_dir, "split_indices.pkl"), "wb") as f:
        pickle.dump(split_info, f)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "models_metrics.json")
    
    # Load existing metrics if file exists
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    
    # Add this model's metrics
    all_metrics.append({
        "model_id": model_id,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_mae": val_metrics["mae"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Write updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"\n===== Model {model_id} Training Complete =====")
    print(f"Model saved to {model_dir}")
    
    return model_dir

if __name__ == "__main__":
    args = parse_arguments()
    
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    all_files = X_train_21 + X_valid_21
    all_century_labels = y_train_21 + y_valid_21
    all_decade_labels = y_train_22 + y_valid_22
    
    full_dataset = MCFileBasedDataset(
        file_paths=all_files,
        path=texts_path,
        century_labels=all_century_labels,
        decade_labels=all_decade_labels,
        tokenizer=tokenizer,
        max_length=1536
    )
    
    print(f"Full dataset size: {len(full_dataset)}")
    
    model_dir = train_model_with_id(args.model_id, full_dataset)
    
    print(f"Training complete. Model saved to {model_dir}")