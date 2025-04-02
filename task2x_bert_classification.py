import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
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

class ChunkedFileBasedMultiTaskDataset(Dataset):
    def __init__(self, file_paths, path, century_labels, decade_labels, tokenizer, chunk_size=512, stride=256, max_chunks=5):
        self.file_paths = file_paths
        self.path = path
        self.century_labels = century_labels
        self.decade_labels = decade_labels
        self.tokenizer = tokenizer
        self.chunk_size = min(chunk_size, 512)  # Ensure we don't exceed BERT's limit
        self.stride = min(stride, self.chunk_size // 2)  # Make sure stride is reasonable
        self.max_chunks = max_chunks
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        century_label = self.century_labels[idx]
        decade_label = self.decade_labels[idx]
        
        with open(os.path.join(self.path, file_path), 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Instead of manually tokenizing, use the tokenizer's built-in chunking capability
        # This will handle the maximum sequence length properly
        chunks = []
        
        # Process the text in smaller segments to avoid potential memory issues
        max_text_length = 100000  # Process at most 100k characters at a time
        for i in range(0, len(text), max_text_length):
            text_segment = text[i:i+max_text_length]
            
            # Use the tokenizer to chunk and encode the text segment
            encodings = self.tokenizer(
                text_segment,
                truncation=True,
                padding='max_length',
                max_length=self.chunk_size,
                return_overflowing_tokens=True,  # Enable chunking
                stride=self.stride,
                return_tensors='pt'
            )
            
            # Add each chunk to our list
            for j in range(encodings['input_ids'].shape[0]):
                if len(chunks) >= self.max_chunks:
                    break
                
                chunks.append({
                    'input_ids': encodings['input_ids'][j],
                    'attention_mask': encodings['attention_mask'][j]
                })
                
            if len(chunks) >= self.max_chunks:
                break
        
        # If there are no chunks (possible but very unlikely), create a simple one
        if not chunks:
            simple_encoding = self.tokenizer(
                text[:self.chunk_size * 4],  # Take just the beginning of text
                truncation=True,
                padding='max_length',
                max_length=self.chunk_size,
                return_tensors='pt'
            )
            chunks.append({
                'input_ids': simple_encoding['input_ids'][0],
                'attention_mask': simple_encoding['attention_mask'][0]
            })
            
        # Create batch tensors for all chunks
        batch_input_ids = torch.stack([chunk['input_ids'] for chunk in chunks])
        batch_attention_mask = torch.stack([chunk['attention_mask'] for chunk in chunks])
        
        # Pad to max_chunks if needed
        if len(chunks) < self.max_chunks:
            pad_size = self.max_chunks - len(chunks)
            pad_input_ids = torch.full((pad_size, self.chunk_size), 
                                      self.tokenizer.pad_token_id, 
                                      dtype=torch.long)
            pad_attention_mask = torch.zeros(pad_size, self.chunk_size, dtype=torch.long)
            
            batch_input_ids = torch.cat([batch_input_ids, pad_input_ids], dim=0)
            batch_attention_mask = torch.cat([batch_attention_mask, pad_attention_mask], dim=0)
        
        # Ensure we never exceed max_chunks or chunk_size
        batch_input_ids = batch_input_ids[:self.max_chunks, :self.chunk_size]
        batch_attention_mask = batch_attention_mask[:self.max_chunks, :self.chunk_size]
        
        # Return a dictionary with all chunks and labels
        return {
            'input_ids': batch_input_ids,  # Shape: [max_chunks, chunk_size]
            'attention_mask': batch_attention_mask,  # Shape: [max_chunks, chunk_size]
            'century_labels': torch.tensor(century_label, dtype=torch.long),
            'decade_labels': torch.tensor(decade_label, dtype=torch.long),
            'num_chunks': torch.tensor(min(len(chunks), self.max_chunks), dtype=torch.long)
        }

# Use the chunked dataset instead of the original
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train2x = ChunkedFileBasedMultiTaskDataset(
    X_train_21, train_path, y_train_21, y_train_22, tokenizer,
    chunk_size=512,  # Explicitly set to BERT's limit
    stride=200,
    max_chunks=5
)
valid2x = ChunkedFileBasedMultiTaskDataset(
    X_valid_21, valid_path, y_valid_21, y_valid_22, tokenizer,
    chunk_size=512,  # Explicitly set to BERT's limit
    stride=200,
    max_chunks=5
)

# Smaller batch size due to multiple chunks per document
batch_size = 8

train2x_dataloader = DataLoader(train2x, batch_size=batch_size, shuffle=True)
valid2x_dataloader = DataLoader(valid2x, batch_size=batch_size, shuffle=False)

class MultiChunkBertModel(nn.Module):
    def __init__(self, max_chunks=5):
        super(MultiChunkBertModel, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.max_chunks = max_chunks
        
        # Chunk attention layer to combine multiple chunk embeddings
        self.chunk_attention = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification heads
        self.century_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )
        
        self.decade_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 43)
        )
    
    def forward(self, input_ids, attention_mask, num_chunks=None):
        batch_size = input_ids.shape[0]
        
        # Reshape to process all chunks together
        chunk_batch_size = batch_size * self.max_chunks
        input_ids = input_ids.view(chunk_batch_size, -1)
        attention_mask = attention_mask.view(chunk_batch_size, -1)
        
        # Process all chunks through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [chunk_batch_size, hidden_size]
        
        # Reshape back to separate documents and their chunks
        pooled_output = pooled_output.view(batch_size, self.max_chunks, -1)  # [batch_size, max_chunks, hidden_size]
        
        # Calculate attention weights over chunks
        attention_scores = self.chunk_attention(pooled_output)  # [batch_size, max_chunks, 1]
        
        # Create chunk mask based on actual number of chunks
        if num_chunks is not None:
            chunk_mask = torch.arange(self.max_chunks, device=input_ids.device)[None, :] < num_chunks[:, None]
            chunk_mask = chunk_mask.unsqueeze(-1).float()  # [batch_size, max_chunks, 1]
            attention_scores = attention_scores.masked_fill(~chunk_mask.bool(), -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, max_chunks, 1]
        
        # Apply attention to get document representation
        doc_representation = torch.sum(attention_weights * pooled_output, dim=1)  # [batch_size, hidden_size]
        
        # Apply classification heads
        century_logits = self.century_classifier(doc_representation)
        decade_logits = self.decade_classifier(doc_representation)
        
        return century_logits, decade_logits

# Update the evaluate function to handle chunked inputs
def evaluate(model, dataloader):
    model.eval()
    century_correct = 0
    decade_correct = 0
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
            century_logits, decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'],
                num_chunks=batch['num_chunks']
            )
            
            # Calculate loss
            loss = multi_task_loss(
                century_logits, decade_logits, 
                batch['century_labels'], batch['decade_labels']
            )
            total_loss += loss.item()
            batch_count += 1
            
            # Century predictions (classification)
            century_predictions = torch.argmax(century_logits, dim=1)
            century_labels = batch['century_labels']
            century_correct += (century_predictions == century_labels).sum().item()
            
            # Decade predictions (classification)
            combined_labels = (batch['century_labels']*10 + batch['decade_labels'])
            decade_predictions = torch.argmax(decade_logits, dim=1)
            decade_correct += (decade_predictions == combined_labels).sum().item()
            
            # Store predictions and labels
            total_predictions += len(century_predictions)
            century_preds.extend(century_predictions.cpu().numpy())
            century_labels_list.extend(century_labels.cpu().numpy())
            decade_preds.extend(decade_predictions.cpu().numpy())
            decade_labels_list.extend(combined_labels.cpu().numpy())
            
            # For final rank computation
            combined_gt_values.extend(combined_labels.cpu().numpy())
            combined_pred_values.extend(decade_predictions.cpu().numpy())
    
    # Calculate metrics
    average_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    century_accuracy = century_correct / total_predictions
    decade_accuracy = decade_correct / total_predictions
    century_mae = mean_avg_error(np.array(century_labels_list), np.array(century_preds))
    
    # For decade classification, calculate MAE between true and predicted class
    decade_mae = mean_avg_error(np.array(decade_labels_list), np.array(decade_preds))
    
    # Calculate Final Rank (FR)
    final_rank = mean_avg_error(np.array(combined_gt_values), np.array(combined_pred_values))
    
    return average_loss, century_accuracy, century_mae, decade_accuracy, decade_mae, final_rank

def multi_task_loss(century_logits, decade_logits, 
                    century_labels, decade_labels, 
                    century_weight=0.3, decade_weight=0.7):
    century_loss = nn.CrossEntropyLoss()(century_logits, century_labels)
    
    # Calculate combined century-decade labels (0-42) for classification
    combined_labels = (century_labels*10 + decade_labels)
    decade_loss = nn.CrossEntropyLoss()(decade_logits, combined_labels)

    return century_weight * century_loss + decade_weight * decade_loss

def mean_avg_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def train_multi_task_model(model, train_loader, val_loader, epochs=10):
    model_save_dir = 'models/task2x_bert/'  # Updated save directory for BERT models
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
            
            century_logits, decade_logits = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'],
                num_chunks=batch['num_chunks']
            )
            
            loss = multi_task_loss(
                century_logits, decade_logits, 
                batch['century_labels'], batch['decade_labels']
            )
            
            loss.backward()
            optimizer.step()
        
        # Evaluate using the unified function
        val_loss, val_century_accuracy, val_century_mae, val_decade_accuracy, val_decade_mae, val_final_rank = evaluate(model, val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            # Save the best model (so far) to disk
            model_path = os.path.join(model_save_dir, f'best_model_epoch_{epoch+1}_classification_bert.pt')  # Updated filename
            torch.save(best_model_state, model_path)
            
            print(f"Epoch {epoch+1}: New best model saved with validation loss: {val_loss:.4f}, saved to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: No improvement for {epochs_no_improve} epochs.")
            
            # Check for early stopping
            if epochs_no_improve >= early_stopping_tolerance:
                print(f"Early stopping triggered after {epoch+1} epochs with no improvement for {early_stopping_tolerance} epochs.")
                break
        
        print(f"Epoch {epoch+1}: Century Accuracy = {val_century_accuracy:.4f}, Century MAE = {val_century_mae:.4f}, " 
              f"Decade Accuracy = {val_decade_accuracy:.4f}, Decade MAE = {val_decade_mae:.4f}, "
              f"Final Rank = {val_final_rank:.4f}, Val Loss = {val_loss:.4f}")
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save the final best model
        final_model_path = os.path.join(model_save_dir, 'best_model_classification_bert.pt')  # Updated filename
        torch.save(best_model_state, final_model_path)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}, saved to {final_model_path}")
    
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiChunkBertModel()  # Changed to multi-chunk model
    model.to(device)
    
    print("Training multi-chunk BERT model...")
    trained_model = train_multi_task_model(model, train2x_dataloader, valid2x_dataloader, epochs=10)
    
    print("Training complete.")