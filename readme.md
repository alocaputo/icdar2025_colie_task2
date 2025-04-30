# ICDAR 2025 COLIE Task 2

This repository contains team UniUD-DUTH-AUEB's submission for the ICDAR 2025 Competition on Automatic Classification of Literary Epochs (COLIE) Task 2, which focuses on temporal text classification of historical documents.

## Overview

[Task 2](https://www.kaggle.com/competitions/icdar-2025-ColiE_Task2/) aims to automatically predict when a historical document was written, with two subtasks:
- Task 2.1: Century classification (15th to 19th century)
- Task 2.2: Decade classification within each century

Our approach combines both tasks using a regression-based model that predicts the absolute decade directly.

## Repository Structure

- `task2x_combined_regression.py`: Main training script for the regression model
- `new_test_eval.py`: Evaluation script for individual models on test data
- `ensemble_test_eval.py`: Evaluation script for ensemble models on test data
- `tables.py`: Script to generate performance tables and visualizations
- `run_exepriment.sh`: Bash script to run experiments

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- pandas
- numpy
- scikit-learn
- tqdm
- matplotlib
- seaborn

## Installation

```bash
# Clone the repository
git clone https://github.com/alocaputo/icdar2025_colie_task2.git
cd icdar2025_colie_task2

# Install dependencies
pip install torch transformers pandas numpy tqdm scikit-learn matplotlib seaborn
```

## Data Structure

The data should be organized as follows:

```
data/
└── Task2/
    ├── texts/       # Text files
    │   ├── train/
    │   ├── valid/
    │   └── test/
    ├── task2.1/     # Century labels
    │   ├── train.csv
    │   └── valid.csv
    └── task2.2/     # Decade labels
        ├── train.csv
        └── valid.csv
```

## Usage

### Training the regression model

```bash
python task2x_combined_regression.py
```

This will train a Longformer-based regression model to predict the absolute decade directly.

### Evaluating on test data

For a single model evaluation:

```bash
python new_test_eval.py --model_type regression --model_path models/best_single_head_regression_model_epoch_4.pt --rounding floor
```

Parameters:
- `--model_type`: Type of model (regression or classification)
- `--model_path`: Path to the model checkpoint
- `--rounding`: Method for rounding regression outputs (round or floor)

### Running ensemble evaluation

```bash
python ensemble_test_eval.py --model_path models/ --ensemble_info best_ensemble_info.pkl
```

Parameters:
- `--model_path`: Directory containing model checkpoints
- `--ensemble_info`: Path to save/load ensemble configuration

### Running all experiments

```bash
bash run_exepriment.sh
```

## Approach

Our approach uses the Longformer model to handle long document inputs, with the following key features:

1. **Combined Task Formulation**: We reformulate the century and decade classification as a single regression task predicting the absolute decade.
2. **Regression-Based Prediction**: The model predicts a continuous value representing the absolute decade, which is then converted back to century and decade.
3. **Ensemble Methods**: We implement both simple averaging and weighted ensembles of models from different training epochs.
4. **Rounding Strategies**: We explore different rounding strategies (floor vs. round) to convert continuous predictions to discrete decades.

## Model Architecture

The model consists of:
- A Longformer base model (`allenai/longformer-base-4096`)
- A single-head regression approach with:
  - Hidden layer of size 512
  - ReLU activation and dropout (0.3)
  - Final linear layer for decade prediction

## Results

The model performance is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

We implement two ensemble approaches:
1. Simple Average: Equal weighting of all model predictions
2. Weighted Average: Models weighted inversely to their validation MAE

The `tables.py` script generates the tables that are included in the final report, showing:
- Performance across different models
- Ensemble performance comparison

### Performance Results

The following metrics were computed on a 40% subset of the validation data:

| Model | MAE (round) | MAE (floor) | MSE (round) | MSE (floor) |
|-------|-------------|-------------|-------------|-------------|
| Regressor | 1.330 ± 0.036 | 1.129 ± 0.040 | 6.589 ± 0.856 | 6.233 ± 0.846 |
| Ensemble (Average) | 1.255 ± 0.035 | 1.154 ± 0.044 | 6.308 ± 0.829 | 6.113 ± 0.811 |
| Ensemble (Weighted) | 1.254 ± 0.035 | 1.153 ± 0.043 | 6.305 ± 0.831 | 6.113 ± 0.811 |

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{UniUD-DUTH-AUEB2025icdar,
  title={xxx},
  author={Locaputo, A., Paparrigopoulou, A. and Platanou, P.},
  booktitle={xxx},
  year={2025}
}
```
