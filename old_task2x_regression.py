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

class FileBasedMultiTaskDataset(Dataset):
    def __init__(self, file_paths, path, century_labels, decade_labels, tokenizer, max_length=1536):
        self.file_paths = file_paths            # file name (.txt)
        self.path = path                        # path (train/valid)
        self.century_labels = century_labels    # data21
        self.decade_labels = decade_labels      # data22
        self.tokenizer = tokenizer
        self.max_length = max_length            # default 512*3 = 1536
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        century_label = self.century_labels[idx]
        decade_label = self.decade_labels[idx]
        
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
        
        encoding['century_labels'] = torch.tensor(century_label, dtype=torch.long)
        encoding['decade_labels'] = torch.tensor(decade_label, dtype=torch.long)
        
        return encoding

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
train2x = FileBasedMultiTaskDataset(X_train_21, train_path, y_train_21, y_train_22, tokenizer)
valid2x = FileBasedMultiTaskDataset(X_valid_21, valid_path, y_valid_21, y_valid_22, tokenizer)

batch_size = 16

train2x_dataloader = DataLoader(train2x, batch_size=batch_size, shuffle=True)
valid2x_dataloader = DataLoader(valid2x, batch_size=batch_size, shuffle=False)

class MultiTaskLongformerModel(nn.Module):
    def __init__(self):
        super(MultiTaskLongformerModel, self).__init__()
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        # Century head (classification)
        self.century_classifier = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )
        
        # Decade head (regression) - modified to output values between 0 and 49
        self.decade_regressor = nn.Sequential(
            nn.Linear(self.longformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Constrains output between 0 and 1
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        century_logits = self.century_classifier(pooled_output)
        decade_prediction = self.decade_regressor(pooled_output).squeeze(-1) * 42
        
        return century_logits, decade_prediction


def multi_task_loss(century_logits, decade_predictions, 
                    century_labels, decade_labels, 
                    century_weight=0.3, decade_weight=0.7):
    century_loss = nn.CrossEntropyLoss()(century_logits, century_labels)
    # Calculate normalized_decade_labels as century*10 + decade
    normalized_decade_labels = (century_labels*10 + decade_labels).float()
    decade_loss = nn.MSELoss()(decade_predictions, normalized_decade_labels)

    return century_weight * century_loss + decade_weight * decade_loss

def mean_avg_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def evaluate(model, dataloader):
    model.eval()
    century_correct = 0
    total_predictions = 0
    century_preds = []
    century_labels_list = []
    decade_preds = []
    decade_labels_list = []
    # For calculating Final Rank
    combined_gt_values = []
    combined_pred_values = []
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            century_logits, decade_predictions = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            # Calculate loss
            loss = multi_task_loss(
                century_logits, decade_predictions, 
                batch['century_labels'], batch['decade_labels']
            )
            total_loss += loss.item()
            batch_count += 1
            
            # Century predictions (classification)
            century_predictions = torch.argmax(century_logits, dim=1)
            century_labels = batch['century_labels']
            century_correct += (century_predictions == century_labels).sum().item()
            
            # Store predictions and labels
            total_predictions += len(century_predictions)
            century_preds.extend(century_predictions.cpu().numpy())
            century_labels_list.extend(century_labels.cpu().numpy())
            decade_preds.extend(decade_predictions.cpu().numpy())
            
            # Calculate combined century and decade values for evaluation
            combined_gt = (batch['century_labels']*10 + batch['decade_labels']).cpu().numpy()
            combined_pred = (century_predictions*10 + decade_predictions).cpu().numpy()
            
            decade_labels_list.extend(combined_gt)
            combined_gt_values.extend(combined_gt)
            combined_pred_values.extend(combined_pred)
    
    # Calculate metrics
    average_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    century_accuracy = century_correct / total_predictions
    century_mae = mean_avg_error(np.array(century_labels_list), np.array(century_preds))
    
    # For decade regression, we use MAE as the metric
    decade_mae = mean_avg_error(np.array(decade_labels_list), np.array(decade_preds))
    
    # Calculate Final Rank (FR)
    final_rank = mean_avg_error(np.array(combined_gt_values), np.array(combined_pred_values))
    
    return average_loss, century_accuracy, century_mae, None, decade_mae, final_rank

def train_multi_task_model(model, train_loader, val_loader, epochs=10):
    model_save_dir = 'models/task2x/'
    os.makedirs(model_save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    early_stopping_tolerance = 3
    
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            century_logits, decade_predictions = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            loss = multi_task_loss(
                century_logits, decade_predictions, 
                batch['century_labels'], batch['decade_labels']
            )
            
            loss.backward()
            optimizer.step()
        
        # Evaluate using the unified function
        val_loss, val_century_accuracy, val_century_mae, _, val_decade_mae, val_final_rank = evaluate(model, val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            # Save the best model (so far) to disk
            model_path = os.path.join(model_save_dir, f'best_model_epoch_{epoch+1}_regression.pt')
            torch.save(best_model_state, model_path)
            
            print(f"Epoch {epoch+1}: New best model saved with validation loss: {val_loss:.4f}, saved to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: No improvement for {epochs_no_improve} epochs.")
            
            # Check for early stopping
            if epochs_no_improve >= early_stopping_tolerance:
                print(f"Early stopping triggered after {epoch+1} epochs with no improvement for {early_stopping_tolerance} epochs.")
                break
        
        print(f"Epoch {epoch+1}: Century Accuracy = {val_century_accuracy:.4f}, Century MAE = {val_century_mae:.4f}, Decade MAE = {val_decade_mae:.4f}, Final Rank = {val_final_rank:.4f}, Val Loss = {val_loss:.4f}")
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save the final best model
        final_model_path = os.path.join(model_save_dir, 'best_model_regression.pt')
        torch.save(best_model_state, final_model_path)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}, saved to {final_model_path}")
    
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskLongformerModel()
    model.to(device)
    
    print("Training multi-task model...")
    trained_model = train_multi_task_model(model, train2x_dataloader, valid2x_dataloader, epochs=10)
    
    print("Training complete.")