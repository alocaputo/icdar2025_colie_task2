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

# train_path = os.path.join(texts_path, 'train')
# valid_path = os.path.join(texts_path, 'valid')

# train_files = os.listdir(train_path)
# valid_files = os.listdir(valid_path)

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

print(f"Train: {len(train21)}, Test: {len(valid21)}")
# with open('blacklist.pkl', 'rb') as f:
#     blacklist = pickle.load(f)

# blacklist_train = blacklist['train']
# blacklist_valid = blacklist['valid']

X_train_21 = [] # file names (.txt)
y_train_21 = [] 

X_valid_21 = [] # file names (.txt)
y_valid_21 = []

for idx, row in train21.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()
    if 'project' in text.lower() and 'gutenberg' in text.lower():
        continue

    # if idx in blacklist_train:
    #     continue

    X_train_21.append(file_name)
    y_train_21.append(century-1)

for idx, row in valid21.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()
    if 'project' in text.lower() and 'gutenberg' in text.lower():
        continue

    # if idx in blacklist_valid:
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
    # if idx in blacklist_train:
    #     continue

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()
    if 'project' in text.lower() and 'gutenberg' in text.lower():
        continue

    # if idx in blacklist_train:
    #     continue

    X_train_22.append(file_name)
    y_train_22.append(century-1)

for idx, row in valid22.iterrows():
    file_name = row.file_name
    century = row.century
    # if idx in blacklist_valid:
    #     continue

    with open(os.path.join(texts_path, file_name), 'r') as file:
        text = file.read()
    if 'project' in text.lower() and 'gutenberg' in text.lower():
        continue
    
    # if idx in blacklist_valid:
    #     continue

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
        
        # Don't use buffer to avoid device issues
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, logits, targets):
        # Get the device from logits
        device = logits.device
        
        # Apply softmax to get probabilities
        probs = self.smax(logits)
        
        # Calculate negative entropy term (for regularization)
        log_vals = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
        neg_entropy = torch.sum(torch.sum(probs * log_vals, dim=1))
        
        # Calculate annealed entropy weight
        if self.annealing:
            current_entropy_weight = max(0, (self.initial_temperature * 
                                            (1 - self.current_epoch / self.annealing_epochs)))
        else:
            current_entropy_weight = self.entropy_weight
        
        # Distance penalty term
        # For each sample, calculate weighted absolute difference between each class value and true value
        batch_size = targets.size(0)
        
        # Create range tensor directly on the target device
        range_vals = torch.arange(0, self.num_classes, dtype=torch.float, device=device)
        
        # Make sure targets is on the correct device
        targets = targets.to(device)
        
        # Create the expanded tensors directly on the device
        expanded_targets = targets.view(-1, 1).expand(batch_size, self.num_classes).float()
        expanded_range = range_vals.view(1, -1).expand(batch_size, -1)
        
        # print(f"Targets device: {targets.device}, Range device: {range_vals.device}")
        # print(f"Expanded targets device: {expanded_targets.device}, Expanded range device: {expanded_range.device}")
        
        # Calculate distance penalty with power q
        distance_penalty = torch.pow(torch.abs(expanded_range - expanded_targets), self.q)
        weighted_penalty = torch.sum(probs * distance_penalty)
        
        # Combine losses
        loss = (neg_entropy * current_entropy_weight) + (self.loss_weight * weighted_penalty)
        
        return loss

class SingleHeadLongformerModel(nn.Module):
    def __init__(self):
        super(SingleHeadLongformerModel, self).__init__()
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        # Classification head - outputs logits for 43 classes (decades)
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

def evaluate_single_head(model, dataloader, criterion):
    model.eval()
    total_predictions = 0
    preds = []
    labels_list = []
    total_loss = 0
    batch_count = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            labels = batch['decade_labels']
            
            # Calculate loss using our custom criterion
            loss = criterion(decade_logits, labels)
            total_loss += loss.item()
            batch_count += 1
            
            # Get predicted class
            probs = nn.Softmax(dim=1)(decade_logits)
            _, predicted_classes = torch.max(probs, dim=1)
            
            # Weighted prediction using probabilities
            # Calculate expected value: sum(class_value * probability)
            class_values = torch.arange(0, 43, device=device, dtype=torch.float)
            weighted_preds = torch.sum(probs * class_values.unsqueeze(0), dim=1)
            
            # Count exact matches
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += len(labels)
            
            # Store predictions and labels for metrics calculation
            preds.extend(predicted_classes.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    # Calculate metrics
    average_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate MAE for the predictions
    mae = mean_avg_error(np.array(labels_list), np.array(preds))
    
    return average_loss, accuracy, mae

def save_checkpoint(model, optimizer, epoch, best_val_loss, epochs_no_improve, model_save_dir):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
    }
    checkpoint_path = os.path.join(model_save_dir, 'training_checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

def load_checkpoint(model, optimizer, model_save_dir):
    checkpoint_path = os.path.join(model_save_dir, 'training_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        return model, optimizer, 0, float('inf'), 0, False
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    best_val_loss = checkpoint['best_val_loss']
    epochs_no_improve = checkpoint['epochs_no_improve']
    
    print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
    return model, optimizer, start_epoch, best_val_loss, epochs_no_improve, True

def train_single_head_model(model, train_loader, val_loader, epochs=10, resume=True):
    model_save_dir = 'new_models/task2x_ordinal_classification/'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Initialize our custom loss function with no buffer registration
    criterion = OrdinalClassificationLoss(
        num_classes=43, 
        q=1.0,
        entropy_weight=0.1,
        loss_weight=1.0,
        annealing=True,
        annealing_epochs=epochs,
        initial_temperature=1.0
    )
    
    # Move criterion to device
    criterion = criterion.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Initialize training variables
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_tolerance = 3
    epochs_no_improve = 0
    start_epoch = 0
    
    # Try to load checkpoint if resume is True
    if resume:
        model, optimizer, start_epoch, best_val_loss, epochs_no_improve, resumed = load_checkpoint(
            model, optimizer, model_save_dir)
        if resumed and best_val_loss < float('inf'):
            # Load the best model state from the saved checkpoint
            best_model_path = os.path.join(model_save_dir, 'best_ordinal_classification_model.pt')
            if os.path.exists(best_model_path):
                best_model_state = torch.load(best_model_path)
    
    for epoch in range(start_epoch, epochs):
        # Update the epoch counter in the loss function for annealing
        criterion.set_epoch(epoch)
        
        model.train()
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            labels = batch['decade_labels']
            
            loss = criterion(decade_logits, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        val_loss, val_accuracy, val_mae = evaluate_single_head(model, val_loader, criterion)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            model_path = os.path.join(model_save_dir, f'best_ordinal_classification_model_epoch_{epoch+1}.pt')
            torch.save(best_model_state, model_path)
            
            # Also save the best model to our consistent path
            torch.save(best_model_state, os.path.join(model_save_dir, 'best_ordinal_classification_model.pt'))
            
            print(f"Epoch {epoch+1}: New best ordinal classification model saved with validation loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: No improvement for {epochs_no_improve} epochs.")
            
            if epochs_no_improve >= early_stopping_tolerance:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        print(f"Epoch {epoch+1}: Accuracy = {val_accuracy:.4f}, MAE = {val_mae:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, best_val_loss, epochs_no_improve, model_save_dir)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        final_model_path = os.path.join(model_save_dir, 'best_ordinal_classification_model.pt')
        torch.save(best_model_state, final_model_path)
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    train2x = FileBasedSingleTaskDataset(X_train_21, texts_path, y_train_21, y_train_22, tokenizer)
    valid2x = FileBasedSingleTaskDataset(X_valid_21, texts_path, y_valid_21, y_valid_22, tokenizer)
    
    batch_size = 16
    train2x_dataloader = DataLoader(train2x, batch_size=batch_size, shuffle=True)
    valid2x_dataloader = DataLoader(valid2x, batch_size=batch_size, shuffle=False)
    
    model = SingleHeadLongformerModel()
    model.to(device)
    
    print("Training ordinal classification model...")
    trained_model = train_single_head_model(model, train2x_dataloader, valid2x_dataloader, epochs=10, resume=True)
    
    print("Training complete.")