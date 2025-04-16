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

def evaluate_single_head(model, dataloader):
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

def train_single_head_model(model, train_loader, val_loader, epochs=10):
    model_save_dir = 'new_models/task2x/'
    os.makedirs(model_save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_tolerance = 3
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            decade_predictions = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            # Convert labels to float for regression
            labels = batch['decade_labels'].float()
            
            loss = criterion(decade_predictions, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        val_loss, val_mae, val_mse = evaluate_single_head(model, val_loader)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            model_path = os.path.join(model_save_dir, f'best_single_head_regression_model_epoch_{epoch+1}.pt')
            torch.save(best_model_state, model_path)
            
            print(f"Epoch {epoch+1}: New best regression model saved with validation loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: No improvement for {epochs_no_improve} epochs.")
            
            if epochs_no_improve >= early_stopping_tolerance:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        print(f"Epoch {epoch+1}: Final Rank (MAE) = {val_mae:.4f}, MSE = {val_mse:.4f}, Val Loss = {val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        final_model_path = os.path.join(model_save_dir, 'best_single_head_regression_model.pt')
        torch.save(best_model_state, final_model_path)
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    train2x = FileBasedSingleTaskDataset(X_train_21, train_path, y_train_21, y_train_22, tokenizer)
    valid2x = FileBasedSingleTaskDataset(X_valid_21, valid_path, y_valid_21, y_valid_22, tokenizer)
    
    batch_size = 16
    train2x_dataloader = DataLoader(train2x, batch_size=batch_size, shuffle=True)
    valid2x_dataloader = DataLoader(valid2x, batch_size=batch_size, shuffle=False)
    
    model = SingleHeadLongformerModel()
    model.to(device)
    
    print("Training regression model...")
    trained_model = train_single_head_model(model, train2x_dataloader, valid2x_dataloader, epochs=10)
    
    print("Training complete.")