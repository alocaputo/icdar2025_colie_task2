import os
import pandas as pd
import numpy as np
import random

import torch
from transformers import BertTokenizer, BertModel

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

from tqdm import tqdm
import pickle

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
# =        Sentence Embeddings            =
# =========================================

from sentence_transformers import SentenceTransformer

model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# model_name = "sentence-transformers/all-MiniLM-L6-v2"

sent_model = SentenceTransformer(model_name, trust_remote_code=True)
sent_model.max_seq_length = 1500

print("Computing train embeddings")
embeddings = sent_model.encode([open(os.path.join(train_path, file_name), 'r').read() for file_name in X_train], show_progress_bar=True)

save_folder_path = os.path.join('embeddings', model_name.split('/')[-1])

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# Save embeddings
with open(f'{save_folder_path}/train.pkl', 'wb') as f:
    embeddings = dict(zip(X_train, embeddings))
    pickle.dump(embeddings, f)

# print("Computing valid (full) embeddings")

# valid_full_emb = sent_model.encode([open(os.path.join(valid_path, file_name), 'r').read() for file_name in X_valid_full], show_progress_bar=True)
# with open(f'{save_folder_path}/valid_full.pkl', 'wb') as f:
#     valid_full_emb = dict(zip(X_valid_full, valid_full_emb))
#     pickle.dump(valid_full_emb, f)

# print("Computing valid embeddings")
# # valid_emb = sent_model.encode([open(os.path.join(valid_path, file_name), 'r').read() for file_name in X_valid], show_progress_bar=True)

# with open(f'{save_folder_path}/valid.pkl', 'wb') as f:
#     valid_emb = { xv: valid_full_emb[xv] for xv in X_valid}
#     pickle.dump(valid_emb, f)