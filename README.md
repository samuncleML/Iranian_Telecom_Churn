# Iranian Telecom Churn Prediction

A machine learning pipeline for predicting customer churn in the Iranian telecommunications sector. This project implements a deep neural network model trained on customer data to identify customers likely to cancel their subscriptions.

## Project Structure

```
Iranian_Telecom_Churn/
├── data/
│   ├── processed/          # Preprocessed datasets
│   │   ├── data_train.csv
│   │   ├── data_test.csv
│   │   └── data_validation.csv
│   └── raw/               # Original dataset
│       └── Customer Churn.csv
├── models/                # Trained model checkpoints
│   └── Iranian_Telecom_Churn_model.pth
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   └── Iranian_Telecom_Churn_Pipeline.ipynb  # Full pipeline
├── src/                   # Source code
│   ├── __init__.py
│   ├── dataset.py         # PyTorch Dataset class
│   ├── module.py          # Neural network model
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Training script
│   ├── test.py            # Testing script
│   └── test_analysis.py   # Analysis tools
├── main.py                # CLI entry point
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Iranian_Telecom_Churn
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Train the model:
```bash
python main.py train --epochs 20
```

Evaluate the model:
```bash
python main.py evaluate
```

Run full pipeline (train + evaluate):
```bash
python main.py full
```

## Cross-Validation

The project includes a 4-fold cross-validation workflow to improve model robustness and reduce sensitivity to a single train/validation split.

- Implemented in `src/cross_val.py`
- Uses the four partition files in `data/cross_val_data/`
- Each fold trains on three partitions and validates on the remaining partition
- Produces cross-validated validation metrics across all four folds
- Helps estimate model performance more reliably than a single hold-out set

### Jupyter Notebooks

Open the notebooks for interactive exploration:

- [EDA.ipynb](notebooks/EDA.ipynb) - Exploratory Data Analysis with visualizations
- [Iranian_Telecom_Churn_Pipeline.ipynb](notebooks/Iranian_Telecom_Churn_Pipeline.ipynb) - Complete end-to-end pipeline

## Model Architecture

The model is a deep neural network implemented in PyTorch (see [src/module.py](src/module.py)). It uses a compact encoder-style architecture with two wide hidden layers and batch normalization.

### Network Layers

The model takes 13 input features (all features including 'Status' and 'Complains'). Total trainable parameters: **138,811**.

| Layer | Input | Output | Activation | BatchNorm |
|-------|-------|--------|------------|-----------|
| layer1 | 13 | 30 | LeakyReLU | - |
| layer2 | 30 | 1024 | LeakyReLU | Yes |
| layer3 | 1024 | 50 | LeakyReLU | Yes |
| layer4 | 50 | 1024 | LeakyReLU | - |
| layer5 | 1024 | 1 | Sigmoid | - |

### Design Choices

- **LeakyReLU activation**: Used after dense layers to maintain gradient flow for negative inputs
- **Sigmoid activation**: Used at the output for binary churn probability prediction
- **BatchNorm**: Applied after the two intermediate wide layers to stabilize training
- **Wide hidden layer**: The network expands to 1024 units in a bottleneck-style architecture for richer feature representation

## Data Preprocessing

The pipeline applies different transformations based on feature distributions. The preprocessing is implemented in [src/preprocess.py](src/preprocess.py) and can also be run interactively in the [EDA.ipynb](notebooks/EDA.ipynb) notebook.

### Feature Groups and Transformations

The original dataset has 14 columns. All features including 'Status' and 'Complains' are kept, resulting in 13 features for the model.

| Scaler | Features | Rationale |
|--------|----------|-----------|
| **MinMaxScaler** | Tariff Plan, Status, Complains | Binary/categorical with bounded values (0 or 1) |
| **StandardScaler** | Age Group, Age | Features following a normal/Gaussian distribution |
| **PowerTransformer** | Call Failure, Subscription Length, Charge Amount, Seconds of Use, Frequency of use, Frequency of SMS, Distinct Called Numbers, Customer Value | Right-skewed distributions; PowerTransformer applies Yeo-Johnson transformation to make data more Gaussian-like |
| **Dropped** | None | All features are used in the model |

### Preprocessing Pipeline

1. **Load raw data**: Read from `data/raw/Customer Churn.csv`
2. **Keep all features**: All 13 features including 'Status' and 'Complains' are retained
3. **Split data**: Train (70%) / Validation (15%) / Test (15%) using stratified sampling
4. **Apply transformations**:
   - Fit transformers on training data only
   - Apply same transformations to validation and test data
5. **Save processed data**: Export to `data/processed/` directory

### Transformation Details

- **MinMaxScaler**: Scales features to [0, 1] range. Used for binary/categorical features where the relative distribution is already uniform.
- **StandardScaler**: Standardizes features by removing mean and scaling to unit variance (z-score normalization). Used for features that follow a normal distribution.
- **PowerTransformer**: Applies Yeo-Johnson transformation to make data more Gaussian-like. This is particularly effective for right-skewed features common in telecom data (e.g., call duration, charges).

## Training Configuration

- **Optimizer**: Adam
- **Initial Learning Rate**: 0.001
- **Scheduler**: StepLR (gamma=0.02, step_size=7) - Learning rate is multiplied by 0.02 every 7 epochs
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Batch Size**: 32
- **Train/Validation/Test Split**: 70% / 15% / 15%

### Training Process

The training loop (implemented in [src/train.py](src/train.py)) performs:

1. **Forward pass**: Compute model predictions
2. **Loss calculation**: BCE loss between predictions and ground truth
3. **Backward pass**: Compute gradients
4. **Optimization step**: Update model weights using Adam optimizer
5. **Learning rate scheduling**: StepLR reduces LR by factor of 0.02 after every 7 epochs
6. **Metrics computation**: Track accuracy, F1 score, recall, and precision

The model is trained for multiple epochs with both training and validation metrics logged at each epoch to monitor overfitting.

### Dataset Features

The dataset contains 14 original columns. All features are kept (including 'Status' and 'Complains'), resulting in **13 input features** and 1 target variable (Churn).

#### Original Dataset Columns (14 total)
| Feature | Description | Type |
|---------|-------------|------|
| Call Failure | Number of call failures | Numeric |
| Subscription Length | Length of subscription (months) | Numeric |
| Charge Amount | Amount charged | Numeric |
| Seconds of Use | Total seconds of use | Numeric |
| Frequency of use | How often the service is used | Numeric |
| Frequency of SMS | SMS usage frequency | Numeric |
| Distinct Called Numbers | Number of unique contacts | Numeric |
| Age Group | Customer's age group | Numeric |
| Tariff Plan | Customer's tariff plan | Binary (0/1) |
| Age | Customer's age | Numeric |
| Customer Value | Calculated customer value score | Numeric |
| Status | Customer status | Binary |
| Complains | Customer complaints | Binary |
| Churn | Target: 1 = churned, 0 = retained | Binary |

#### Model Input Features (13 features)
The following features are used as model inputs after preprocessing:

| Feature | Description | Transformation |
|---------|-------------|----------------|
| Tariff Plan | Customer's tariff plan | MinMaxScaler |
| Age Group | Customer's age group | StandardScaler |
| Age | Customer's age | StandardScaler |
| Call Failure | Number of call failures | PowerTransformer |
| Subscription Length | Length of subscription (months) | PowerTransformer |
| Charge Amount | Amount charged | PowerTransformer |
| Seconds of Use | Total seconds of use | PowerTransformer |
| Frequency of use | How often the service is used | PowerTransformer |
| Frequency of SMS | SMS usage frequency | PowerTransformer |
| Distinct Called Numbers | Number of unique contacts | PowerTransformer |
| Customer Value | Calculated customer value score | PowerTransformer |
| Status | Customer status | MinMaxScaler |
| Complains | Customer complaints | MinMaxScaler |


## Evaluation Metrics

- Accuracy
- F1 Score
- Recall
- Precision


## Model Performance Visualization

![ROC Curve and Confusion Matrix](notebooks/ROC_Confusion_Matrix.png)

### Model Evaluation: ROC Curve & Confusion Matrix

The figure above summarizes the model's performance on the test set:

- **ROC Curve (Receiver Operating Characteristic):**
   - Plots the True Positive Rate (TPR) vs. False Positive Rate (FPR) at various thresholds.
   - The Area Under the Curve (AUC) quantifies the model's ability to distinguish between churn and non-churn customers. An AUC close to 1.0 indicates excellent discrimination.

- **Confusion Matrix:**
   - Shows the counts of true positives, true negatives, false positives, and false negatives.
   - Helps visualize the model's accuracy and the types of errors it makes.

In this project, the model achieves a high AUC and demonstrates strong classification performance, as seen in the confusion matrix. This indicates the model is effective at identifying customers who are likely to churn.

### Test Set Performance Metrics

| Metric     | Value    |
|------------|----------|
| Accuracy   |  98.41 % |
| F1 Score   |  94.52 % |
| Recall     |  93.24 % |
| Precision  |  95.83 % |
| ROC AUC    | ~100.0 % |

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- torchmetrics

## License

This project is provided for educational and research purposes.

## Acknowledgments

Dataset source: Iranian Telecom Customer Churn Dataset (on Kaggle)