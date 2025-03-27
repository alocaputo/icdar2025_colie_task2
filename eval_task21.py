import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import ftfy
import pickle
import argparse

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



texts_path = './data/Task2/texts'

valid_path = os.path.join(texts_path, 'valid')

valid_files = os.listdir(valid_path)

valid21 = pd.read_csv('./data/Task2/task2.1/valid.csv')

valid21.rename(columns={'label': 'century'}, inplace=True)
valid21['file_name'] = valid21['id']
valid21['id'] = valid21.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
valid21.set_index('id', inplace=True)

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
    

# Update argparse implementation to use flags
parser = argparse.ArgumentParser(description='Evaluate ICDAR 2025 Task 2.1 models')
parser.add_argument('--blacklist', action='store_true', help='Use model with blacklist')
parser.add_argument('--equal', action='store_true', help='Use model with equal weighting')
parser.add_argument('--exclude-blacklist', action='store_true', help='Exclude blacklisted IDs during validation')

args = parser.parse_args()

with open('blacklist.pkl', 'rb') as f:
    blacklist = pickle.load(f)

blacklist_valid = blacklist['valid']

X_valid_21 = []
y_valid_21 = []

for idx, row in valid21.iterrows():
    file_name = row.file_name
    century = row.century
    if args.exclude_blacklist and idx in blacklist_valid:
        continue
    X_valid_21.append(file_name)
    y_valid_21.append(century-1)


valid21_dataset = TextDataset(X_valid_21, y_valid_21, valid_path)

batch_size = 16
valid21_dataloader = DataLoader(valid21_dataset, batch_size=batch_size, shuffle=False)

# Construct model name based on flags
model_name = "century_classifier_weights_3"
if args.equal:
    model_name += "_equal"
if args.blacklist:
    model_name += "_blacklist"

model_century_classifier = CenturyClassifier(model, 5)
model_path = f'models/task21/{model_name}.pt'
model_century_classifier.load_state_dict(torch.load(model_path))
model_century_classifier.to(device)
model_century_classifier.eval()

def evaluate_longformer(model, tokenizer, dataloader, device, mode='argmax'):

    y_pred = []
    y_true = []
    
    if mode == 'argmax':
        with torch.no_grad():
            for text, labels in tqdm(dataloader):
                tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=512*3).to(device)
                output = model(**tokens)
                predictions = output.argmax(dim=1)
                y_pred.extend(predictions.detach().cpu().numpy())
                y_true.extend(labels.cpu().numpy())
    else:
        raise NotImplementedError
        
    return y_pred, y_true

y_pred, y_true = evaluate_longformer(model_century_classifier, tokenizer, valid21_dataloader, device, mode='argmax')

# Save the results to pickle file
results = {
    'predictions': y_pred,
    'true_labels': y_true,
    'model_name': model_name
}

# Create directory structure if it doesn't exist
if args.exclude_blacklist:
    save_dir = os.path.join('results', 'task21', 'blacklist')
else:
    save_dir = os.path.join('results', 'task21', 'full')
os.makedirs(save_dir, exist_ok=True)

# Save to the specified directory with model_name.pkl as filename
output_path = os.path.join(save_dir, f"{model_name}.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(results, f)
    
print(f"Evaluation completed. Results saved to {output_path}")