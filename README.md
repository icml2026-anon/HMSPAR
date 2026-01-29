# HMSPAR: Homologous Multimodal Fusion with Parallel Sparsity–Dynamics Awareness for Sparse Sequence Classiffcation

A comprehensive multi-modal machine learning framework for time series classification and prediction tasks in retail and e-commerce domains.A comprehensive multi-modal machine learning framework for time series classification and prediction tasks in retail and e-commerce domains.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Pre-trained Models](#pre-trained-models)
- [Dataset Generation](#dataset-generation)
- [Usage](#usage)
  - [Main HMSPAR Model](#main-hmspar-model)
  - [Baseline Models](#baseline-models)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Supported Tasks](#supported-tasks)

## Overview

HMSPAR is a state-of-the-art multi-modal deep learning framework designed for retail analytics and time series classification. The model integrates multiple data modalities including:HMSPAR is a state-of-the-art multi-modal deep learning framework designed for retail analytics and time series classification. The model integrates multiple data modalities including:

- **Time Series Data**: Sequential transaction patterns, sales trends: Sequential transaction patterns, sales trends
- **Text Embeddings**: Product descriptions, customer behavior descriptions: Product descriptions, customer behavior descriptions
- **Image Data**: ISA-GAF (Gramian Angular Field) representations of time series: ISA-GAF (Gramian Angular Field) representations of time series
- **Static Features**: Demographic and categorical information (optional): Demographic and categorical information (optional)

The framework employs advanced techniques including:The framework employs advanced techniques including:
- Taylor-KAN (Kolmogorov-Arnold Network) layers for time series encoding Taylor-KAN (Kolmogorov-Arnold Network) layers for time series encoding
- Progressive Layer Enhancement (PLE) for multi-modal fusion Progressive Layer Enhancement (PLE) for multi-modal fusion
- ResNet-based image encoding for visual time series representations ResNet-based image encoding for visual time series representations
- Sparse parameter sharing across modalities Sparse parameter sharing across modalities

## Features

- **Multi-modal Architecture**: Integrates time series, text, image, and static data: Integrates time series, text, image, and static data
- **Advanced Time Series Encoding**: Taylor-KAN layers for temporal pattern recognition: Taylor-KAN layers for temporal pattern recognition
- **Multi-modal Fusion**: PLE-based fusion with load balancing and expert routing: PLE-based fusion with load balancing and expert routing
- **Comprehensive Baselines**: 12+ baseline models for comparison: 12+ baseline models for comparison
- **Multiple Dataset Support**: Merchant, Retail, CDNOW, Instacart, TaFeng, Sales Weekly: Merchant, Retail, CDNOW, Instacart, TaFeng, Sales Weekly
- **Flexible Task Support**: Classification, prediction, and risk assessment tasks: Classification, prediction, and risk assessment tasks

## Requirements

See See `requirements.txt` for detailed dependencies. Key requirements: for detailed dependencies. Key requirements:

- Python >= 3.8 Python >= 3.8
- PyTorch >= 2.3.0 PyTorch >= 2.3.0
- scikit-learn >= 1.2.0 scikit-learn >= 1.2.0
- sentence-transformers >= 2.2.0
- pyts >= 0.13.0

## Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd HMSPAR
```

2. **Create virtual environment:**
```bash
python -m venv hmspar_env
source hmspar_env/bin/activate  # On Windows: hmspar_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Pre-trained Models

For offline usage, you need to download the following pre-trained models:

### Sentence Transformers Model

The framework uses sentence transformers for text embedding generation. Download the required model:

```bash
# Create models directory
mkdir -p models/sentence_transformers

# Download the model (choose one of the following methods)

# Method 1: Using Python
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('models/sentence_transformers/all-MiniLM-L6-v2')
"

# Method 2: Manual download from Hugging Face
# Visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# Download all files to: models/sentence_transformers/all-MiniLM-L6-v2/
```

### ResNet Pre-trained Weights

The image encoder uses ResNet-18 with ImageNet pre-trained weights. These will be automatically downloaded by torchvision when first used, or you can pre-download them:

```bash
python -c "
import torchvision.models as models
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
"
```

### TabPFN Model (Optional)

If using TabPFN baseline, download the pre-trained model:

```bash
python -c "
from tabpfn import TabPFNClassifier
classifier = TabPFNClassifier()  # This will download the model
"
```

## Dataset Generation

**Note**: We do not provide external datasets. Use our data generation scripts to create simulated datasets locally by running `data/data_generator.py`.

### Generate Simulated Data

1. **Generate base datasets:**
```bash
python data/data_generator.py
```

2. **Convert to multi-modal format:**
```bash
python data/modality_converter.py --dataset all
```

### Real Dataset Sources

If you want to use the original real datasets instead of simulations, you can download them from the following sources:

- **Online Retail**: [UCI ML Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **CDNOW**: [CDNOW Customer Data](https://www.brucehardie.com/datasets/) - Available from Bruce Hardie's datasets
- **Instacart**: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data) - Kaggle competition data
- **Ta-Feng**: [Ta-Feng Grocery Dataset](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset) - Available on Kaggle
- **Sales Weekly**: Based on [Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only/data) or similar retail sales datasets

**Note**: When using real datasets, you may need to modify the data preprocessing pipeline in `data/modality_converter.py` to match the actual data schema and format.

## Usage

### Main HMSPAR Model

#### Training on Merchant Dataset (Multi-industry)

```bash
# Train on specific industry
python train.py --industry Industry-0 --epochs 100 --batch-size 32

# Available industries: Industry-0, Industry-1, Industry-2, Industry-3
```

#### Training on Specific Datasets

```bash
# Retail dataset
python train_retail.py --task churn --epochs 50 --learning-rate 0.001

# CDNOW dataset
python train_cdnow.py --task engagement --batch-size 64 --epochs 75

# Instacart dataset
python train_instacart.py --task churn --learning-rate 0.0005

# TaFeng dataset
python train_tafeng.py --task repurchase --epochs 80

# Sales Weekly dataset
python train_sales_weekly.py --task seasonality --batch-size 48
```

#### Model Evaluation

```bash
# Evaluate trained model
python evaluate.py --industry Industry-0
```

### Baseline Models

The framework includes comprehensive baseline comparisons:

#### Tree-based Models

```bash
# XGBoost
python baselines/xgboost_baseline.py --dataset merchant --industry Industry-0

# LightGBM  
python baselines/lightgbm_baseline.py --dataset retail

# CatBoost
python baselines/catboost_baseline.py --dataset cdnow

# Random Forest
python baselines/randomforest_baseline.py --dataset instacart
```

#### Deep Learning Models

```bash
# LSTM
python baselines/lstm_baseline.py --dataset merchant --industry Industry-1

# CNN
python baselines/cnn_baseline.py --dataset retail --task churn

# Time-MoE
python baselines/timemoe_baseline.py --dataset sales_weekly

# MPTSNet
python baselines/mptsnet_baseline.py --dataset tafeng

# UniTS
python baselines/units_baseline.py --dataset merchant --industry Industry-2

# VQShape
python baselines/vqshape_baseline.py --dataset cdnow
```

#### Advanced Baselines

```bash
# TabPFN (Pre-trained Transformer)
python baselines/tabpfn_baseline.py --dataset instacart

# TabICL (In-Context Learning)
python baselines/tabicl_baseline.py --dataset merchant --industry Industry-3

# T-Loss (Temporal Loss)
python baselines/tloss_baseline.py --dataset sales_weekly --model-type advanced

# Modern Neural Categorical Analysis
python baselines/modernnca_baseline.py --dataset retail
```

#### Batch Baseline Execution

```bash
# Run all baselines for a dataset
python baselines/run_all_baselines.sh merchant Industry-0
```

### Training Parameters

#### HMSPAR Model Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--learning-rate` | Learning rate | 0.001 | 0.0001-0.01 |
| `--batch-size` | Batch size | 32 | 16-128 |
| `--epochs` | Training epochs | 100 | 50-200 |
| `--ts-hidden-dim` | Time series hidden dimension | 128 | 64-512 |
| `--fusion-dim` | Fusion layer dimension | 256 | 128-1024 |
| `--dropout-rate` | Dropout rate | 0.3 | 0.1-0.5 |
| `--order` | Taylor expansion order | 4 | 2-8 |
| `--n-experts` | Number of experts | 4 | 2-8 |
| `--n-heads` | Attention heads | 4 | 2-8 |

## Project Structure

```
HMSPAR/
├── configs/                 # Configuration files
│   ├── config.py           # Main configuration
│   └── __init__.py
├── data/                   # Data processing modules
│   ├── data_generator.py   # Simulated data generation
│   ├── modality_converter.py  # Multi-modal data conversion
│   ├── dataset.py          # Dataset classes
│   └── appendix_tasks.py   # Additional task definitions
├── models/                 # Model architectures
│   ├── hmspar.py          # Main HMSPAR model
│   └── __init__.py
├── utils/                  # Utility functions
│   ├── metrics.py         # Evaluation metrics
│   ├── seed.py           # Random seed utilities
│   ├── focal_loss.py     # Focal loss implementation
│   └── __init__.py
├── baselines/             # Baseline model implementations
│   ├── xgboost_baseline.py
│   ├── lightgbm_baseline.py
│   ├── lstm_baseline.py
│   ├── cnn_baseline.py
│   ├── tabpfn_baseline.py
│   ├── timemoe_baseline.py
│   ├── mptsnet_baseline.py
│   ├── units_baseline.py
│   ├── vqshape_baseline.py
│   ├── tabicl_baseline.py
│   ├── tloss_baseline.py
│   ├── modernnca_baseline.py
│   ├── catboost_baseline.py
│   ├── randomforest_baseline.py
│   ├── data_utils.py
│   └── run_all_baselines.sh
├── scripts/               # Automation scripts
│   ├── 01_generate_data.sh
│   ├── 02_convert_modalities.sh
│   ├── 03_train_industry0.sh
│   ├── 03_train_industry1.sh
│   ├── 03_train_industry2.sh
│   ├── 03_train_industry3.sh
│   └── 04_evaluate_all.sh
├── train*.py             # Dataset-specific training scripts
├── evaluate.py           # Model evaluation
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Model Architecture

### HMSPAR Components

1. **Time Series Encoder**: Taylor-KAN layers with adaptive pooling
2. **Image Encoder**: ResNet-18 with custom input adaptation
3. **Text Processor**: Linear projection of sentence embeddings
4. **Static Feature Encoder**: Multi-layer neural network (optional)
5. **PLE Fusion**: Progressive Layer Enhancement with expert routing
6. **Prediction Head**: Multi-layer classifier with dropout

### Key Components

- **Taylor-KAN**: Uses Taylor series expansions for time series encoding
- **PLE Fusion**: Separates shared and modality-specific expert networks
- **Load Balancing**: Balances expert utilization during training
- **Multi-scale Processing**: Handles different time series lengths and frequencies

## Supported Tasks

### Classification Tasks
- **Customer Churn Prediction**: Identify at-risk customers
- **Engagement Classification**: Categorize customer activity levels  
- **Risk Assessment**: Classify customer risk profiles

### Prediction Tasks
- **Repurchase Prediction**: Forecast customer return behavior
- **Seasonality Detection**: Identify seasonal patterns in sales
- **Sales Risk Classification**: Assess product performance risk

### Dataset-Task Mapping

| Dataset | Supported Tasks |
|---------|----------------|
| Merchant | All classification tasks (per industry) |
| Retail | Churn prediction, engagement classification |
| CDNOW | Churn prediction, engagement classification |
| Instacart | Churn prediction, engagement classification |
| TaFeng | Repurchase prediction, risk classification |
| Sales Weekly | Seasonality detection, sales risk classification |

## Performance

Performance varies by dataset complexity and task difficulty. Baseline models provide comparison benchmarks.

---

This is an anonymous research implementation. All datasets are simulated and generated using the provided data generation scripts.
