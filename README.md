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
│   ├── raw/               # Original dataset
│   │   └── Customer Churn.csv
│   └── cross_val_data/    # Four cross-validation partitions
├── models/                # Trained model checkpoints
│   └── Iranian_Telecom_Churn_model.pth
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   ├── Iranian_Telecom_Churn_Pipeline.ipynb  # Full pipeline
│   └── ROC_Confusion_Matrix.md
├── src/                   # Source code
│   ├── __init__.py
│   ├── cross_val.py       # Cross-validation workflow
│   ├── dataset.py         # PyTorch Dataset class and data loaders
│   ├── module.py          # Neural network model
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Training utilities
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

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
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

The CLI supports custom model save/load paths, so you can run the pipeline on different machines without changing code.

## Windows Compatibility

This pipeline is compatible with every Windows machine. It uses flexible relative path handling across the repository, including os.path.join, os.path.dirname(__file__), and configurable model and data paths. That means the model pipeline can run reliably on Windows without hard-coded file locations.

## Cross-Validation

The project includes a 4-fold cross-validation workflow to improve model robustness and reduce sensitivity to a single train/validation split.

- Implemented in src/cross_val.py
- Uses the four partition files in data/cross_val_data/
- Each fold trains on three partitions and validates on the remaining partition
- Produces robust cross-validated validation metrics

## Jupyter Notebooks

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

The pipeline applies different transformations based on feature distributions. The preprocessing is implemented in [src/preprocess.py](src/preprocess.py) and can also be explored in [notebooks/EDA.ipynb](notebooks/EDA.ipynb).

### Feature Groups and Transformations

The original dataset has 14 columns. All features including 'Status' and 'Complains' are kept, resulting in 13 model inputs.

| Scaler | Features | Rationale |
|--------|----------|-----------|
| **MinMaxScaler** | Tariff Plan, Status, Complains | Binary/categorical with bounded values |
| **StandardScaler** | Age Group, Age | Features that resemble a normal distribution |
| **PowerTransformer** | Call Failure, Subscription Length, Charge Amount, Seconds of Use, Frequency of use, Frequency of SMS, Distinct Called Numbers, Customer Value | Right-skewed numeric features benefit from Yeo-Johnson transformation |
| **Dropped** | None | All available features are used in the model |

### Preprocessing Pipeline

1. **Load raw data**: Read from data/raw/Customer Churn.csv
2. **Keep all features**: All 13 model inputs including 'Status' and 'Complains' are retained
3. **Split data**: Train (70%) / Validation (15%) / Test (15%) using stratified sampling
4. **Transform data**:
   - Fit transformers on training data only
   - Apply identical transforms to validation and test sets
5. **Save processed data**: Export to data/processed/

### Transformation Details

- **MinMaxScaler**: Scales binary/categorical fields to [0, 1]
- **StandardScaler**: Standardizes continuous features by removing the mean and scaling to unit variance
- **PowerTransformer**: Applies a Yeo-Johnson transformation to reduce skew and make numeric data more Gaussian-like

## Training Configuration

- **Optimizer**: Adam
- **Initial Learning Rate**: 0.001
- **Scheduler**: StepLR (gamma=0.02, step_size=7)
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Batch Size**: 32
- **Train/Validation/Test Split**: 70% / 15% / 15%

### Training Process

The training loop in [src/train.py](src/train.py) performs:

1. **Forward pass**: Compute predictions
2. **Loss calculation**: BCE loss between predictions and labels
3. **Backward pass**: Compute gradients
4. **Optimization step**: Update model weights with Adam
5. **Learning rate scheduling**: Reduce the learning rate every 7 epochs
6. **Metrics computation**: Track accuracy, F1, recall, and precision

## Evaluation Metrics

- Accuracy
- F1 Score
- Recall
- Precision

## Model Performance Visualization

<img width="1954" height="975" alt="image" src="https://github.com/user-attachments/assets/2508767a-ffdf-4858-bbd6-7624402e6f62" />

### Model Evaluation: ROC Curve & Confusion Matrix

The chart and confusion matrix summarize the model's performance on the test set:

- **ROC Curve**: Measures the model's ability to distinguish churn vs retained customers
- **Confusion Matrix**: Visualizes true/false positives and negatives

### Test Set Performance Metrics

| Metric     | Value    |
|------------|----------|
| Accuracy   |  98.41 % |
| F1 Score   |  94.52 % |
| Recall     |  93.24 % |
| Precision  |  95.83 % |
| ROC AUC    |  99.02 % |

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
