import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel, LongformerTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestDataset(Dataset):
    def __init__(self, file_paths, path, tokenizer, max_length=1536, is_validation=False, df=None):
        self.file_paths = file_paths
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_validation = is_validation
        self.df = df
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_id = file_path  # Keep the original filename for reference
        
        with open(os.path.join(self.path, file_path), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['file_id'] = file_id
        
        # Add ground truth labels for validation data
        if self.is_validation and self.df is not None:
            # Extract the ID from filename (assuming format like valid_textXXX.txt)
            file_name = file_path
            encoding['century_label'] = self.df.loc[self.df['file_name'] == file_name, 'century'].values[0] - 1  # Convert to 0-indexed
            
        return encoding


if __name__ == "__main__":
    # Path to test files
    test_path = './data/Task2/test'
    test_files = sorted(os.listdir(test_path))
    
    print(f"Test files found: {len(test_files)}")
    
    # Initialize tokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    # Create test dataset
    test_dataset = TestDataset(test_files, test_path, tokenizer)
    
    # Analyze token counts, file sizes and word counts
    token_counts = []
    file_sizes = []
    word_counts = []
    
    print("Analyzing text files in test set...")
    for file_path in tqdm(test_files):
        full_path = os.path.join(test_path, file_path)
        
        # Get file size in KB
        file_size = os.path.getsize(full_path) / 1024
        file_sizes.append(file_size)
        
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Count words (rough approximation by splitting on whitespace)
        words = text.split()
        word_counts.append(len(words))
        
        # Count tokens without truncation
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        token_counts.append(len(tokens))
    
    # Calculate token statistics
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    avg_tokens = sum(token_counts) / len(token_counts)
    median_tokens = np.median(token_counts)
    
    # Calculate file size statistics
    max_file_size = max(file_sizes)
    min_file_size = min(file_sizes)
    avg_file_size = sum(file_sizes) / len(file_sizes)
    median_file_size = np.median(file_sizes)
    
    # Calculate word count statistics
    max_words = max(word_counts)
    min_words = min(word_counts)
    avg_words = sum(word_counts) / len(word_counts)
    median_words = np.median(word_counts)
    
    # Calculate additional statistics
    std_tokens = np.std(token_counts)
    percentile_25 = np.percentile(token_counts, 25)
    percentile_75 = np.percentile(token_counts, 75)
    
    # Print token statistics
    print(f"\nToken count statistics for test set:")
    print(f"Maximum tokens: {max_tokens}")
    print(f"Minimum tokens: {min_tokens}")
    print(f"Average tokens: {avg_tokens:.2f}")
    print(f"Median tokens: {median_tokens}")
    print(f"Standard deviation: {std_tokens:.2f}")
    print(f"25th percentile: {percentile_25}")
    print(f"75th percentile: {percentile_75}")
    
    # Print file size statistics
    print(f"\nFile size statistics (in KB):")
    print(f"Maximum file size: {max_file_size:.2f} KB")
    print(f"Minimum file size: {min_file_size:.2f} KB")
    print(f"Average file size: {avg_file_size:.2f} KB")
    print(f"Median file size: {median_file_size:.2f} KB")
    
    # Print word count statistics
    print(f"\nWord count statistics:")
    print(f"Maximum words: {max_words}")
    print(f"Minimum words: {min_words}")
    print(f"Average words: {avg_words:.2f}")
    print(f"Median words: {median_words}")
    
    # Distribution by token count ranges
    print("\nDistribution of token counts:")
    ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 750), (750, 1000), (1000, 1500), (1500, 2000), (2000, 4000), (4000, float('inf'))]
    for lower, upper in ranges:
        count = sum(lower <= count < upper for count in token_counts)
        percentage = count / len(token_counts) * 100
        print(f"Files with {lower}-{upper if upper != float('inf') else '+'} tokens: {count} ({percentage:.2f}%)")
    
    # Count files below common thresholds
    print(f"\nFiles below certain thresholds:")
    print(f"Files below 100 tokens: {sum(count < 100 for count in token_counts)} ({sum(count < 100 for count in token_counts)/len(token_counts)*100:.2f}%)")
    print(f"Files below 200 tokens: {sum(count < 200 for count in token_counts)} ({sum(count < 200 for count in token_counts)/len(token_counts)*100:.2f}%)")
    print(f"Files below 300 tokens: {sum(count < 300 for count in token_counts)} ({sum(count < 300 for count in token_counts)/len(token_counts)*100:.2f}%)")
    
    # Count files exceeding common thresholds
    print(f"\nFiles exceeding common thresholds:")
    print(f"Files exceeding 512 tokens: {sum(count > 512 for count in token_counts)} ({sum(count > 512 for count in token_counts)/len(token_counts)*100:.2f}%)")
    print(f"Files exceeding 1024 tokens: {sum(count > 1024 for count in token_counts)} ({sum(count > 1024 for count in token_counts)/len(token_counts)*100:.2f}%)")
    print(f"Files exceeding 1536 tokens: {sum(count > 1536 for count in token_counts)} ({sum(count > 1536 for count in token_counts)/len(token_counts)*100:.2f}%)")
    print(f"Files exceeding 2048 tokens: {sum(count > 2048 for count in token_counts)} ({sum(count > 2048 for count in token_counts)/len(token_counts)*100:.2f}%)")
    print(f"Files exceeding 4096 tokens: {sum(count > 4096 for count in token_counts)} ({sum(count > 4096 for count in token_counts)/len(token_counts)*100:.2f}%)")

