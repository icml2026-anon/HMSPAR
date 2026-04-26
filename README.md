# HMSPAR

**HMSPAR** (**H**omologous **M**ulti-modal **S**parse **P**attern **A**nomaly **R**ecognition) is the first framework to synergize three homologous modalities — raw time series, ISA-GAF image representations, and semantic text embeddings — for sparse sequence classification.

---

## Architecture

| Component | Description |
|-----------|-------------|
| **Taylor-KAN Encoder** | Time series encoder using Taylor expansion-based KAN layers for non-linear feature extraction |
| **ISA-GAF Image Encoder** | Dual-channel (Instantaneous Sum of Angles + Gramian Angular Field) images encoded with a modified ResNet18 |
| **Text Projector** | Multilingual sentence embeddings linearly projected into the fusion space |
| **PLE Fusion** | Progressive Layered Extraction with Top-*K* sparse MoE routing across modality-specific and shared experts; load balancing via auxiliary loss |

```
Time Series ──► Taylor-KAN Encoder ──┐
                                      ├──► PLE-MoE Fusion ──► Classification Head
ISA-GAF Image ──► ResNet18 ──────────┤
                                      │
Text Embedding ──► Linear Projection ─┘
```

---

## Datasets

TaFeng is included under `data/tafeng/`. All other datasets must be downloaded and preprocessed separately (see dataset instructions below).

| Dataset | Task | Script |
|---------|------|--------|
| **TaFeng** *(included)* | Repurchase prediction | `train_tafeng.py` |
| Simulated Merchant | Anomaly detection | `train_hmspar.py` |
| CDNOW | Churn prediction | `train_cdnow.py` |
| Instacart | Purchase prediction | `train_instacart.py` |
| Online Retail | Churn prediction | `train_retail.py` |
| Sales Weekly | Seasonality detection | `train_sales_weekly.py` |

### Data Directory Structure

```
data/
├── tafeng/
│   ├── ta_feng_all_months_merged.csv
│   └── repurchase_task/
│       ├── amount_series.npy
│       ├── trans_series.npy
│       ├── isa_gaf_images.npy
│       ├── text_embeddings.npy
│       └── labels.npy
├── cdnow/              # download separately
├── instacart/          # download separately
├── retail/             # download separately
└── sales_weekly/       # download separately
```

---

## Baselines

13 competitive baselines organized by representational strategy:

| Category | Models |
|----------|--------|
| Temporal-centric | xLSTM, ModernTCN, Hydra, SoftShape, DSN, MPTSNet, TimeMoE |
| Tabular-oriented | TabM, TabICL, TabPFN |
| Transformation & Foundation Models | TimeLLM, GPT4TS, GAF-CNN |

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.8 and PyTorch ≥ 2.0. A CUDA-capable GPU is recommended.

---

## Quick Start

### TaFeng (dataset included)

```bash
python train_tafeng.py --multi-seed
```

### Simulated Merchant Dataset

Generate the dataset first:

```bash
python data/data_generator.py
```

Train on each industry segment:

```bash
python train_hmspar.py --industry Industry-0 --multi-seed
python train_hmspar.py --industry Industry-1 --multi-seed
python train_hmspar.py --industry Industry-2 --multi-seed
python train_hmspar.py --industry Industry-3 --multi-seed
```

### Other Datasets

```bash
python train_cdnow.py --multi-seed
python train_instacart.py --multi-seed
python train_retail.py --multi-seed
python train_sales_weekly.py --multi-seed
```

### Run Baselines

```bash
cd baselines
python xlstm_baseline.py    --dataset merchant --industry Industry-0 --multi-seed
python mptsnet_baseline.py  --dataset merchant --industry Industry-0 --multi-seed
python softshape_baseline.py --dataset merchant --industry Industry-0 --multi-seed
# Other baselines follow the same interface
```

---

## Evaluation Protocol

All experiments use **3 random seeds** (42, 123, 456) and report **mean ± std** for Accuracy, Precision, Recall, F1-Score, AUC-ROC, and AUPRC on the held-out test set.

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

```
torch>=2.0
torchvision
transformers
sentence-transformers
scikit-learn
pandas
numpy
tabpfn
```

---

## License

This project is released under the [MIT License](LICENSE).
