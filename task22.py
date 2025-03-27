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
parser.add_argument('--freeze', action='store_true', help='Freeze the base model parameters')
parser.add_argument('--exclude-blacklist', action='store_true', help='Exclude blacklisted IDs during traing')
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

X_train_22 = []
y_train_22 = []

X_valid_22 = []
y_valid_22 = []

for idx, row in train22.iterrows():
    file_name = row.file_name
    century = row.century
    if args.exclude_blacklist and idx in blacklist_train:
        continue
    X_train_22.append(file_name)
    y_train_22.append(century-1)

for idx, row in valid22.iterrows():
    file_name = row.file_name
    century = row.century
    if args.exclude_blacklist and idx in blacklist_valid:
        continue
    X_valid_22.append(file_name)
    y_valid_22.append(century-1)
    

# Same number for each class
np.random.seed(RANDOM_SEED)

train_size =  min(Counter(y_train_22).values())
valid_size =  min(Counter(y_valid_22).values())    
    
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
    
centuy_model_path = f"models/task21/century_classifier_weights_3"
if args.equal:
    centuy_model_path += "_equal"
if args.blacklist:
    centuy_model_path += "_blacklist"
centuy_model_path += ".pt"

model_century_classifier = CenturyClassifier(model, 5)
model_century_classifier.load_state_dict(torch.load(centuy_model_path))
model_century_classifier.to(device)

class DecadeClassifier(nn.Module):
    def __init__(self, century_model, num_decades=10):
        super(DecadeClassifier, self).__init__()
        # Get the pretrained longformer from the century classifier
        self.bert = century_model.bert
        
        if args.freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            
        self.dropout = nn.Dropout(0.1)
        
        # Create classifier for decades
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_decades)
        
        # Add a connection from century prediction to decade prediction
        self.century_to_decade = nn.Linear(5, num_decades)
        
    def forward(self, input_ids, attention_mask, century_logits=None, token_type_ids=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        decade_logits = self.classifier(pooled_output)
        
        if century_logits is not None:
            century_contribution = self.century_to_decade(century_logits)
            decade_logits = decade_logits + century_contribution
            
        return decade_logits
    
model_decade_classifier = DecadeClassifier(model_century_classifier)
model_decade_classifier.to(device)

print("Using all available samples", '[Excluding blacklist]' if args.blacklist else '')
train_dataset = TextDataset(X_train_22, y_train_22, train_path)
valid_dataset = TextDataset(X_valid_22, y_valid_22, valid_path)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print(f"Train: {len(train_dataset)}, Test: {len(valid_dataset)}")
print("Training...")

optimizer = optim.Adam(model_decade_classifier.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 3
model_decade_classifier.train()

# Finetune the model ========================================================================================
for epoch in range(EPOCHS):
    for text, labels in tqdm(train_dataloader):
        tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=512*3,).to(device)
        labels = torch.tensor(labels).to(device)
        
        # Create labels_encoded directly on the correct device
        batch_size = labels.size(0)
        labels_encoded = torch.zeros(batch_size, 10, device=device)
        # Use scatter_ for proper one-hot encoding
        labels_encoded.scatter_(1, (labels).unsqueeze(1), 1)

        with torch.no_grad():
                model_century_classifier.eval()
                century_logits = model_century_classifier(**tokens)

        optimizer.zero_grad()
        decade_logits = model_decade_classifier(**tokens, century_logits=century_logits)
        #loss = criterion(output, labels)
        loss = loss_fn(decade_logits, labels_encoded)
        loss.backward()
        optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())

# Save model with configuration info in filename
model_path = f"models/task22"
if not os.path.exists(model_path):
    os.makedirs(model_path)

model_filename = f"decade_classifier_weights_{EPOCHS}"
if args.exclude_blacklist:
    model_filename += "_blacklist"
model_filename += ".pt"

torch.save(model_decade_classifier.state_dict(), os.path.join(model_path, model_filename))