from torch.utils.data import DataLoader, Dataset
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import ftfy
import pickle
import argparse

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Train a century classifier')
parser.add_argument('--equal', action='store_true', help='Use equal number of samples per class')
parser.add_argument('--blacklist', action='store_true', help='Filter out blacklisted files')
args = parser.parse_args()

# Set random seed
RANDOM_SEED = 42

# Load data

texts_path = './data/Task2/texts'

train_path = os.path.join(texts_path, 'train')
valid_path = os.path.join(texts_path, 'valid')

train_files = os.listdir(train_path)
valid_files = os.listdir(valid_path)

print(f"Train: {len(train_files)}, Test: {len(valid_files)}")

label2century = {1: '17th century', 2: '18th century', 3: '19th century', 4: '20th century', 5: '21st century'}

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

with open('blacklist.pkl', 'rb') as f:
    blacklist = pickle.load(f)

blacklist_train = blacklist['train']
blacklist_valid = blacklist['valid']

from torch.utils.data import DataLoader, Dataset
import ftfy

def get_clean_text(path):
    
    with open(path, 'rb') as infile:
        txt_ry = infile.read()
    misdecoded_text = txt_ry.decode("utf-8", errors="ignore")

    return ftfy.fix_text(misdecoded_text)

class TextDataset(Dataset):
    def __init__(self, file_names, labels, path, clear=False):
        self.file_names = file_names
        self.labels = labels
        self.path = path
        self.clear = clear

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.clear:
            text = get_clean_text(os.path.join(self.path, file_name))
        else:
            with open(os.path.join(self.path, file_name), 'r') as file:
                text = file.read()

        return text, self.labels[idx]
    
# All dataset 

X_train_21 = []
y_train_21 = []

X_valid_21 = []
y_valid_21 = []

for idx, row in train21.iterrows():
    file_name = row.file_name
    century = row.century

    with open(os.path.join(train_path, file_name), 'r') as file:
        text = file.read()
    if 'gutenberg' in text.lower():
        continue

    if args.blacklist and idx in blacklist_train:
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

    if args.blacklist and idx in blacklist_valid:
        continue
    X_valid_21.append(file_name)
    y_valid_21.append(century-1)
    

# Same number for each class
np.random.seed(RANDOM_SEED)

train_size =  min(Counter(y_train_21).values())
valid_size =  min(Counter(y_valid_21).values())

X_train_21_eq = []
y_train_21_eq = []
for i in range(1, 6):
    x = train21[train21['century'] == i].sample(train_size)
    X_train_21_eq.extend(x['file_name'])
    y_train_21_eq.extend(x['century']-1)

X_valid_21_eq = []
y_valid_21_eq = []
for i in range(1, 6):
    x = valid21[valid21['century'] == i].sample(valid_size)
    X_valid_21_eq.extend(x['file_name'])
    y_valid_21_eq.extend(x['century']-1)
    
    
def mean_avg_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# Model
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModel.from_pretrained("allenai/longformer-base-4096")

class CenturyClassifier(nn.Module):
    def __init__(self, longformer, num_classes):
        super(CenturyClassifier, self).__init__()
        self.bert = longformer
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(longformer.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
    
model_century_classifier = CenturyClassifier(model, 5)
model_century_classifier.to(device)

# Choose dataset based on --equal flag
if args.equal:
    print("Using equal number of samples per class")
    train_dataset = TextDataset(X_train_21_eq, y_train_21_eq, train_path)
    valid_dataset = TextDataset(X_valid_21_eq, y_valid_21_eq, valid_path)
else:
    print("Using all available samples")
    train_dataset = TextDataset(X_train_21, y_train_21, train_path)
    valid_dataset = TextDataset(X_valid_21, y_valid_21, valid_path)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print(f"Train: {len(train_dataset)}, Test: {len(valid_dataset)}")
print("Training...")

optimizer = optim.Adam(model_century_classifier.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Define evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text, labels in tqdm(dataloader, desc="Validating"):
            tokens = tokenizer(text, padding='max_length', truncation=True, 
                              return_tensors="pt", max_length=512*3).to(device)
            labels = torch.tensor(labels).to(device)
            output = model(**tokens)
            loss = criterion(output.view(-1, 5), labels.view(-1))
            total_loss += loss.item()
            
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    mae = mean_avg_error(np.array(all_labels), np.array(all_preds))
    
    model.train()
    return total_loss / len(dataloader), accuracy, mae

# Training with validation-based early stopping
patience = 3  # Number of epochs to wait for improvement
max_epochs = 10
best_val_loss = float('inf')
best_model_state = None
epochs_without_improvement = 0

model_century_classifier.train()

for epoch in range(max_epochs):
    # Training phase
    model_century_classifier.train()
    train_loss = 0
    
    for text, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs}"):
        optimizer.zero_grad()
        tokens = tokenizer(text, padding='max_length', truncation=True, 
                          return_tensors="pt", max_length=512*3).to(device)
        labels = torch.tensor(labels).to(device)
        output = model_century_classifier(**tokens)
        loss = criterion(output.view(-1, 5), labels.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_dataloader)
    
    # Validation phase
    val_loss, val_acc, val_mae = evaluate(model_century_classifier, valid_dataloader, criterion)
    
    print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MAE: {val_mae:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model_century_classifier.state_dict().copy()
        epochs_without_improvement = 0
        print(f"► New best model saved with validation loss: {val_loss:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"► No improvement for {epochs_without_improvement} epochs")
    
    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# Load best model before saving
if best_model_state is not None:
    model_century_classifier.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

# Save model with configuration info in filename
model_path = f"models/task21"
if not os.path.exists(model_path):
    os.makedirs(model_path)

model_filename = f"century_classifier_best_val"
if args.equal:
    model_filename += "_equal"
if args.blacklist:
    model_filename += "_blacklist"
model_filename += ".pt"

torch.save(model_century_classifier.state_dict(), os.path.join(model_path, model_filename))
print(f"Model saved to {os.path.join(model_path, model_filename)}")