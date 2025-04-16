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

RANDOM_SEED = 42

texts_path = './data/Task2/texts'

## Load the data ==============================================================================

train_path = os.path.join(texts_path, 'train')
valid_path = os.path.join(texts_path, 'valid')

train_files = os.listdir(train_path)
valid_files = os.listdir(valid_path)

f"Train: {len(train_files)}, Test: {len(valid_files)}"

label2century = {1: '17th century', 2: '18th century', 3: '19th century', 4: '20th century', 5: '21st century'}

train21 = pd.read_csv('./data/Task2/task2.1/train.csv')
valid21 = pd.read_csv('./data/Task2/task2.1/valid.csv')

train = pd.merge(train21, train22, on='id')
train.rename(columns={'label_x': 'century', 'label_y': 'decade'}, inplace=True)
train['file_name'] = train['id']
train['id'] = train.id.str.replace('train_text', '').str.replace('.txt', '').astype(int)
train.set_index('id', inplace=True)


# Load the blacklist
# with open('blacklist.pkl', 'rb') as f:
#     blacklist = pickle.load(f)
# 
# blacklist_train = blacklist['train']
# blacklist_valid = blacklist['valid']

np.random.seed(RANDOM_SEED)

X_train_22 = []
y_train_22 = []

X_valid_22 = []
y_valid_22 = []

for idx, row in train.iterrows():
    file_name = row.file_name
    century = row.century
    decade = row.decade
    
    if idx in blacklist_train:
        continue
    # year = (century+16)*100 + (decade-1)*10
    year = (century-1) * 10 + (decade-1)
    X_train_22.append(file_name)
    y_train_22.append(year)

for idx, row in valid.iterrows():
    file_name = row.file_name
    century = row.century
    decade = row.decade
    
    if idx in blacklist_valid:
        continue
    year = (century-1) * 10 + (decade-1)
    y_valid_22.append(year)
    X_valid_22.append(file_name)


## Create the dataset ============================================================================

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

# Train the model =========================================================
# Longformer is better

import torch
import torch.nn as nn
import torch.optim as optim

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert, num_classes):
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

    # def forward(self, input_ids, token_type_ids, attention_mask):
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

train22_dataset = TextDataset(X_train_22, y_train_22, train_path)
valid22_dataset = TextDataset(X_valid_22, y_valid_22, valid_path) # Full
# valid_full_dataset = TextDataset(X_valid_full, y_valid_full, valid_path)

batch_size = 16
train22_dataloader = DataLoader(train22_dataset, batch_size=batch_size, shuffle=True)
valid22_dataloader = DataLoader(valid22_dataset, batch_size=batch_size, shuffle=False)
# valid_full_dataloader = DataLoader(valid_full_dataset, batch_size=batch_size, shuffle=False)

from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_name = 'google-bert/bert-base-cased'
model_name = 'allenai/longformer-base-4096'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# for param in model.parameters():
#     param.requires_grad = False
    
model_decade_classifier = BertForSequenceClassification(model, 43)
model_decade_classifier.to(device)

optimizer = optim.Adam(model_decade_classifier.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

model_folder = model_name.split('/')[1]

EPOCHS = 3

# Finetune the model ========================================================================================
for epoch in range(EPOCHS):
    model_decade_classifier.train()
    for text, labels in tqdm(train22_dataloader):
        optimizer.zero_grad()
        tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=512*3,).to(device)
        labels = torch.tensor(labels).to(device)
        output = model_decade_classifier(**tokens)
        #loss = criterion(output, labels)
        loss = criterion(output.view(-1, 43), labels.view(-1))
        loss.backward()
        optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())
    # bert_confusion(model_decade_classifier, valid_dataloader)

if not os.path.exists(f"models/{model_folder}"):
    os.makedirs(f"models/{model_folder}")
    
torch.save(model_decade_classifier.state_dict(), f"models/{model_folder}/decade_classifier_weights_{EPOCHS}_coherent.pt")

# Continue training from checkpoint =====================================================

# model_decade_classifier.load_state_dict(torch.load(f"models/{model_folder}/decade_classifier_weights_{EPOCHS}.pt"))
# ADDITIONAL_EPOCHS = 1
# TOTAL_EPOCHS = EPOCHS + ADDITIONAL_EPOCHS

# for epoch in range(EPOCHS, TOTAL_EPOCHS):
#     model_decade_classifier.train()
#     for text, labels in tqdm(train22_dataloader):
#         optimizer.zero_grad()
#         tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=512*3).to(device)
#         labels = torch.tensor(labels).to(device)
#         output = model_decade_classifier(**tokens)
#         loss = criterion(output.view(-1, 43), labels.view(-1))
#         loss.backward()
#         optimizer.step()
#     print("Epoch:", epoch, "Loss:", loss.item())

# # Save the model after additional epochs
# torch.save(model_decade_classifier.state_dict(), f"models/{model_folder}/decade_classifier_weights_{TOTAL_EPOCHS}.pt")