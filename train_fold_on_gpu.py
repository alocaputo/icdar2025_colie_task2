import os
import sys
import argparse
import torch
from task2x_regression_ensemble import (
    SingleHeadLongformerModel, FileBasedSingleTaskDataset, train_model_on_fold,
    convert_to_absolute_decade
)
from torch.utils.data import DataLoader
from transformers import LongformerTokenizer
import pickle
import numpy as np

def train_single_fold(fold_num, gpu_id, data_path, k=5, epochs=10):
    """Train a single fold on a specific GPU"""
    # Check available GPUs and set device safely
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            print(f"Warning: Requested GPU {gpu_id} is not available. Only {num_gpus} GPUs available.")
            if num_gpus > 0:
                gpu_id = 0  # Fall back to the first GPU
                print(f"Falling back to GPU {gpu_id}")
                device = torch.device(f"cuda:{gpu_id}")
            else:
                print("No GPUs available. Using CPU.")
                device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    
    # Load prepared data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_final_train = data['X_final_train']
    y_final_train_century = data['y_final_train_century']
    y_final_train_decade = data['y_final_train_decade']
    combined_path = data['combined_path']
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    # Create k-fold splits
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_indices = list(kf.split(X_final_train))
    
    # Get the specific fold's train/val split
    if fold_num > k or fold_num < 1:
        raise ValueError(f"Fold number must be between 1 and {k}")
    
    train_idx, val_idx = fold_indices[fold_num-1]
    
    print(f"\n=== Training Fold {fold_num}/{k} on {device} ===")
    
    # Create dataset for this fold
    X_train_fold = [X_final_train[i] for i in train_idx]
    y_train_century_fold = [y_final_train_century[i] for i in train_idx]
    y_train_decade_fold = [y_final_train_decade[i] for i in train_idx]
    
    X_val_fold = [X_final_train[i] for i in val_idx]
    y_val_century_fold = [y_final_train_century[i] for i in val_idx]
    y_val_decade_fold = [y_final_train_decade[i] for i in val_idx]
    
    # Fix: Correctly prepare file paths for the dataset by mapping them ahead of time
    X_train_fold_paths = [combined_path[file] for file in X_train_fold]
    X_val_fold_paths = [combined_path[file] for file in X_val_fold]
    
    # Create datasets with file paths directly and limit token length to 1536
    train_fold_dataset = FileBasedSingleTaskDataset(
        X_train_fold_paths, "", # Empty string as base path since full paths are already provided
        y_train_century_fold, y_train_decade_fold, tokenizer, max_length=1536
    )
    val_fold_dataset = FileBasedSingleTaskDataset(
        X_val_fold_paths, "",  # Empty string as base path since full paths are already provided
        y_val_century_fold, y_val_decade_fold, tokenizer, max_length=1536
    )
    
    batch_size = 8
    train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize and train a new model for this fold
    fold_model = SingleHeadLongformerModel()
    fold_model = fold_model.to(device)
    
    trained_model, model_path = train_model_on_fold(
        fold_model, train_loader, val_loader, fold_num, epochs=epochs, device=device
    )
    
    print(f"Fold {fold_num} training complete. Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a specific fold on a specific GPU")
    parser.add_argument("--fold", type=int, required=True, help="Fold number (1-based)")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use")
    parser.add_argument("--data", type=str, required=True, help="Path to prepared data pickle file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--k", type=int, default=5, help="Total number of folds")
    
    args = parser.parse_args()
    
    train_single_fold(args.fold, args.gpu, args.data, args.k, args.epochs)
