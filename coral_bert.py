import os
import pandas as pd
import numpy as np
import random

import torch
from transformers import BertTokenizer, BertModel

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

from tqdm import tqdm

RANDOM_SEED = 42

### =========================================
### =        Load Data and Preprocess       =
### =========================================

texts_path = './data/Task2/texts'

train_path = os.path.join(texts_path, 'train')
valid_path = os.path.join(texts_path, 'valid')

train_files = os.listdir(train_path)
valid_files = os.listdir(valid_path)

label2century = {1: '17th century', 2: '18th century', 3: '19th century', 4: '20th century', 5: '21st century'}

train21 = pd.read_csv('./data/Task2/task2.1/train.csv')
valid21 = pd.read_csv('./data/Task2/task2.1/valid.csv')

train22 = pd.read_csv('./data/Task2/task2.2/train.csv')
valid22 = pd.read_csv('./data/Task2/task2.2/valid.csv')

train = pd.merge(train21, train22, on='id')
train.rename(columns={'label_x': 'century', 'label_y': 'decade'}, inplace=True)
train['file_name'] = train['id']
train['id'] = train.id.str.replace('train_text', '').str.replace('.txt', '').astype(int)
train.set_index('id', inplace=True)

valid = pd.merge(valid21, valid22, on='id')
valid.rename(columns={'label_x': 'century', 'label_y': 'decade'}, inplace=True)
valid['file_name'] = valid['id']
valid['id'] = valid.id.str.replace('valid_text', '').str.replace('.txt', '').astype(int)
valid.set_index('id', inplace=True)

np.random.seed(RANDOM_SEED)

X_train = []
y_train = []
for i in range(1, 6):
    x = train21[train21['label'] == i].sample(1834)
    X_train.extend(x['id'])
    y_train.extend(x['label'])

X_valid_full = valid21['id']
y_valid_full = valid21['label']

X_valid = []
y_valid = []
for i in range(1, 6):
    x = valid21[valid21['label'] == i].sample(457)
    X_valid.extend(x['id'])
    y_valid.extend(x['label'])

y_train = [x-1 for x in y_train]
y_valid = [x-1 for x in y_valid]
y_valid_full = [x-1 for x in y_valid_full]


# =========================================
# =             Train model               =
# =========================================

def mean_avg_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

from torch.utils.data import DataLoader, Dataset

def preprocess_data(document):
    stop_words = stopwords.words('english')
 
    texts = [word for word in simple_preprocess(str(document)) if word not in stop_words]
    
    return ' '.join(texts)

class TextDataset(Dataset):
    def __init__(self, file_names, labels, path, preprocess=False):
        self.file_names = file_names
        self.labels = labels
        self.path = path
        self.preprocess = preprocess

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        with open(os.path.join(self.path, file_name), 'r') as file:
            text = file.read()
            if self.preprocess:
                text = preprocess_data(text)

        levels = [1] * self.labels[idx] + [0] * (5-1 - self.labels[idx]) # NUM_CLASSES = 5
        levels = torch.tensor(levels, dtype=torch.float32)
        return text, self.labels[idx], levels

# Model

import torch
import torch.nn as nn
import torch.optim as optim

class CoralBertForSequenceClassification(nn.Module):
    def __init__(self, bert, num_classes):
        super(CoralBertForSequenceClassification, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.num_classes = num_classes
        # self.classifier = nn.Linear(bert.config.hidden_size, num_classes)
        self.fc = nn.Linear(bert.config.hidden_size, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(num_classes-1).float())

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits = self.fc(pooled_output)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas
    
from torch.nn import functional as F
def coral_loss(logits, levels, imp):
    # print(logits.shape, levels.shape, imp.shape)
    # print(logits.device, levels.device, imp.device)
    return (-torch.sum((F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))*imp, dim=1)).mean()


train_dataset = TextDataset(X_train, y_train, train_path)
valid_dataset = TextDataset(X_valid, y_valid, valid_path)
valid_full_dataset = TextDataset(X_valid_full, y_valid_full, valid_path)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
valid_full_dataloader = DataLoader(valid_full_dataset, batch_size=batch_size, shuffle=False)

def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)

    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0), 
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m/torch.max(m)
    return imp

# Train
from torch.nn import DataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
model = BertModel.from_pretrained("google-bert/bert-base-cased")

# for param in model.parameters():
#     param.requires_grad = False
    
# model_classifier = CoralBertForSequenceClassification(model, 5)
# model_classifier.to(device)

# optimizer = optim.Adam(model_classifier.parameters(), lr=5e-5)
# # criterion = nn.CrossEntropyLoss()

# imp = task_importance_weights(torch.tensor(train_dataloader.dataset.labels))[0:4]
# imp = imp.to(device)



# for epoch in range(10):
#     model_classifier.train()
#     for text, labels, levels in tqdm(train_dataloader):
#         optimizer.zero_grad()
#         tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt").to(device)
#         labels = torch.tensor(labels).to(device)
#         logits, probas = model_classifier(**tokens)
#         levels = torch.tensor(levels).to(device)
#         # imp = torch.ones(4, dtype=torch.float).to('cuda')
#         cost = coral_loss(logits, levels, imp)
#         # loss = criterion(output.view(-1, 5), labels.view(-1)) # loss = criterion(output, labels)
#         # loss.backward()
#         cost.backward()
#         optimizer.step()
#     print("Epoch:", epoch, "Loss:", cost.item())
# # bert_confusion(model_classifier, valid_dataloader)

from accelerate import Accelerator
from accelerate.logging import get_logger
import logging

# Initialize the accelerator
accelerator = Accelerator()

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

device = accelerator.device

model_classifier = CoralBertForSequenceClassification(model, 5)
model_classifier.to(device)

optimizer = optim.Adam(model_classifier.parameters(), lr=5e-5)

# imp = torch.tensor(task_importance_weights(torch.tensor(train_dataloader.dataset.labels))[0:4], dtype=torch.float)
imp = task_importance_weights(torch.tensor(train_dataloader.dataset.labels))[0:4]
imp = imp.to(device)

# Prepare everything with accelerator
model_classifier, optimizer, train_dataloader = accelerator.prepare(
    model_classifier, optimizer, train_dataloader
)

total_loss = 0
num_batches = 0

for epoch in range(100):
    model_classifier.train()
    for text, labels, levels in train_dataloader:
        optimizer.zero_grad()
        tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt").to(device)
        # labels = torch.tensor(labels).to(device)
        lables = labels.to(device)
        logits, probas = model_classifier(**tokens)
        # levels = torch.tensor(levels).to(device)
        levels = levels.to(device)
        cost = coral_loss(logits, levels, imp)
        accelerator.backward(cost)
        optimizer.step()
        total_loss += accelerator.gather_for_metrics(cost).mean().item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    # print("Epoch:", epoch, "Loss:", cost.item())
    logger.info(f"Epoch: {epoch}, Loss (avg): {avg_loss}", main_process_only=True)
    # print(f'{accelerator.device} Gather reduced loss: {accelerator.gather_for_metrics(cost.mean())}\n')
    # print(f'{accelerator.device} Gather loss with no reduction: {accelerator.gather_for_metrics(cost)}\n')

    # Validation
    if epoch % 25 == 0: 
        model_classifier.eval()
        y_true = []
        y_pred = []
        for text, labels, levels in tqdm(valid_full_dataloader):
            tokens = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt").to(device)
            # labels = torch.tensor(labels).to(device)
            labels = labels.to(device)
            logits, probas = model_classifier(**tokens)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((probas > 0.5).cpu().numpy().sum(axis=1))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # print("Validation Full:", mean_avg_error(y_true, y_pred))
        logger.info(f"Validation Full (MAE): {mean_avg_error(y_true, y_pred)}", main_process_only=True)   
        
        
# save model

unwrapped_model_classifier = accelerator.unwrap_model(model_classifier)
# unwrapped_model_classifier.save_pretrained(
#     "models/coral_bert",
#     is_main_process=accelerator.is_main_process,
#     save_function=accelerator.save,
# )
torch.save(unwrapped_model_classifier.state_dict(), "models/coral_bert.pth")

# torch.save(model_classifier.state_dict(), 'models/coral_bert.pth')
# model_classifier.save_pretrained('models/coral_bert')

# End training
accelerator.end_training()