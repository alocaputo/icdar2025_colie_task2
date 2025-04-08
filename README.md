# ICDAR 2025 Text Dating Models

This repository contains different neural network architectures for the ICDAR 2025 Text Dating Challenge. The models are designed to predict the century and decade of historical text documents.

## Project Overview

The task involves predicting when a historical text was written at two levels of granularity:
- **Task 2.1**: Century classification (5 classes)
- **Task 2.2**: Decade within century (10 classes)

Combined, these create a 43-class problem spanning from the 17th to 21st centuries. The models use Longformer architecture to handle the long historical texts.

## Model Architectures

### 1. Multi-Task Classification Model (`task2x_classification.py`)

This baseline architecture uses:
- Longformer base model (limited to a context window of 513*2=1536)
- Two separate classification heads:
  - Century head: 5 classes (17th-21st centuries)
  - Decade head: 43 classes (all possible decade values)
- Loss function combines cross-entropy loss from both heads
- Weights: 30% century, 70% decade

### 2. Consistent Multi-Task Model (`task2x_consistent.py`)

Improves on the baseline with:
- Shared intermediate layer (512 dimensions) between century and decade heads
- Century head: 5 classes (17th-21st centuries)
- Decade head: 10 classes (0-9)
- **Consistency loss**: Penalizes illogical century-decade combinations
  - Encourages late decades of one century to correlate with early decades of next century
  - Encourages early decades of one century to correlate with late decades of previous century
- Weights: 33% century, 47% decade, 20% consistency

### 3. Final Rank Optimized Model (`task2x_consistent_fr.py`)

Further enhances the consistent model with:
- Same architecture as the consistent model
- Adds a **timeline loss** component that directly minimizes the final rank error
  - Calculates expected timeline value from predicted probabilities
  - Minimizes L1 distance to ground truth timeline
- Weights: 25% century, 30% decade, 15% consistency, 30% timeline

### 4. Single-Head Combined Model (`task2x_combined.py`)

Simplifies the approach with:
- Single classification head that directly predicts absolute decade (0-42)
- Converts century and decade labels into a single absolute decade value
- Uses standard cross-entropy loss
- Simpler architecture but the prediction should be more coherent

### 5. Regression Model (`task2x_combined_regression.py`)

A variant of the single-head model that uses regression instead of classification:
- Longformer base model (with 1536 token context window)
- Single regression head that directly predicts continuous values for absolute decade
- MSE loss to minimize squared error between predictions and ground truth
- Simpler architecture with potentially better generalization for the timeline prediction task

## Data Preparation

All models use the same data preparation pipeline:
1. Load and preprocess text files from Task 2.1 and 2.2
2. Filter out texts containing "gutenberg" and those in the blacklist
3. Tokenize using LongformerTokenizer with a maximum length of 1536 tokens
4. Create a custom Dataset class that provides both century and decade labels

## Training and Evaluation

Each model implements:
- Early stopping based on validation loss
- Model checkpointing for best performance
- AdamW optimizer with learning rate 2e-5
- Batch size of 16

Evaluation metrics include:
- Century accuracy and MAE
- Decade accuracy and MAE
- Final Rank (FR) - mean absolute error between predicted and true timeline values

## Usage

To train a model:

```bash
python task2x_classification.py  # For baseline classification model
python task2x_consistent.py      # For consistent multi-task model
python task2x_consistent_fr.py   # For final rank optimized model
python task2x_combined.py        # For single-head combined model
python task2x_combined_regression.py  # For regression model
```

## Evaluation

For model evaluation and generating predictions for submission, use the `test_eval.py` script:

```bash
# Evaluate classification model
python test_eval.py --model_type classification --model_path models/task2x/best_model_classification.pt

# Evaluate consistent models
python test_eval.py --model_type consistent --model_path models/task2x_consistent/best_model_consistent.pt

# Evaluate regression models
python3 test_eval.py --model_type regression --model_path models/task2x/best_single_head_regression_model.pt

# Use conformal prediction for uncertainty estimation
python test_eval.py --conformal --alpha 0.1 --pred_method argmax --model_type consistent

# Compare all prediction methods with conformal prediction
python test_eval.py --conformal --alpha 0.1 --pred_method all --model_type consistent
```

### Evaluation Options

- `--model_type`: Type of model architecture (`multitask`, `classification`, `consistent`, `single_head`, or `regression`)
- `--model_path`: Path to the saved model checkpoint
- `--conformal`: Enable conformal prediction for uncertainty estimation
- `--alpha`: Alpha value for conformal prediction (default: 0.1)
- `--pred_method`: Method for selecting predictions:
  - `argmax`: Standard argmax across all classes
  - `set_argmax`: Select the class with highest probability within the conformal set
  - `threshold_max`: Select based on how much each class exceeds its threshold
  - `threshold_max_norm`: Normalize scores by probability
  - `all`: Run all prediction methods and save to separate files

The evaluation script automatically enforces the temporal constraint that 21st century texts cannot have a decade later than 3 (i.e., 2030s).


## Model Selection

- **Classification model**: Baseline approach, best score: 2.69119 (best model)
- **Consistent model**: Better temporal coherence in predictions: 2.53525 ✅ (epoch 3)
- **Final Rank model**: Optimized directly for the evaluation metric: 2.54046 (epoch 5)
- **Combined model**: Simpler architecture, may generalize better: 2.72107 (epoch 1)
- **Combined model (Regression)**: Simpler architecture, with a regression head: **2.49412** ✅

## Requirements

- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- Scikit-learn
- tqdm

## Directory Structure

The models save checkpoints to:
- `models/task2x/` for classification and combined models
- `models/task2x_consistent/` for consistent and final rank models