import os
import pickle
import argparse
from sklearn.model_selection import train_test_split
from task2x_regression_ensemble import (
    X_train_21, X_valid_21, y_train_21, y_valid_21, 
    y_train_22, y_valid_22, train_path, valid_path
)

def prepare_data(output_path, test_size=0.1, seed=42):
    """Prepare data for distributed ensemble training"""
    # Combine train and validation data
    all_files = X_train_21 + X_valid_21
    all_century_labels = y_train_21 + y_valid_21
    all_decade_labels = y_train_22 + y_valid_22
    
    # Use consistent file path for all data
    combined_path = {**{file: os.path.join(train_path, file) for file in X_train_21}, 
                    **{file: os.path.join(valid_path, file) for file in X_valid_21}}
    
    # Create a separate validation set for final evaluation
    X_final_train, X_final_valid, y_final_train_century, y_final_valid_century, y_final_train_decade, y_final_valid_decade = train_test_split(
        all_files, all_century_labels, all_decade_labels, test_size=test_size, random_state=seed
    )
    
    # Save data to pickle file
    data = {
        'X_final_train': X_final_train,
        'X_final_valid': X_final_valid,
        'y_final_train_century': y_final_train_century,
        'y_final_valid_century': y_final_valid_century,
        'y_final_train_decade': y_final_train_decade,
        'y_final_valid_decade': y_final_valid_decade,
        'combined_path': combined_path,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data prepared and saved to {output_path}")
    print(f"Training set size: {len(X_final_train)}")
    print(f"Validation set size: {len(X_final_valid)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for distributed ensemble training")
    parser.add_argument("--output", type=str, default="./data/ensemble_data.pkl", help="Output path for data pickle file")
    parser.add_argument("--test-size", type=float, default=0.1, help="Size of validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    prepare_data(args.output, args.test_size, args.seed)
