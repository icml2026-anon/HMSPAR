# HMSPAR: Homologous Multimodal Fusion with Parallel Sparsity-Dynamics Awareness for Sparse Sequence Classification

> Accepted at **ACM Multimedia 2026** (MM '26)

**Anonymous repository:** https://anonymous.4open.science/r/HMSPAR-7A5C

---

## Abstract

Sparse sequence classification is a fundamental task in critical domains such as financial risk identification and e-commerce churn prediction. Existing methods often struggle with two major challenges — *structural sparsity* and *inter-sample heterogeneity* — which degrade performance in complex real-world scenarios. We propose **HMSPAR**, a homologous multimodal fusion framework that synchronously transforms raw sequences into visual and semantic representations while preserving semantic consistency. At the core of HMSPAR: (1) the **ISA-GAF** dual-channel encoding explicitly decomposes sequences into Trend and Sparsity Channels for parallel sparsity-dynamics awareness; (2) a **Taylor-KAN**-based Higher-Order Temporal Encoder (HOTE) captures high-order temporal dependencies; and (3) a **Decoupled Sparse Modality Fusion (DSMF)** module with sample-specific sparse MoE gating accommodates inter-sample heterogeneity. HMSPAR is the first framework to synergize three homologous modalities for sparse sequence classification.

---

## Architecture

```
Raw Sequence x
    │
    ├──► ISA-GAF (Trend + Sparsity Channel) ──► ResNet-18   ──► h_img
    ├──► HOTE   (Taylor-KAN × 2, GAP)       ──────────────► h_ts
    └──► Text Descriptor ──► Frozen SBERT ──► Linear Proj  ──► h_text
                                                      │
                                          ┌───────────┘
                                          ▼
                              DSMF  (PLE-style MoE, Top-K sparse gating)
                                          │
                                          ▼
                              Fusion MLP ──► Prediction Head ──► ŷ
```

| Module | Role | Complexity |
|--------|------|-----------|
| **ISA-GAF** | Dual-channel image: Trend (interpolated GASF) + Sparsity (bipolar mask GASF) | O(T²) |
| **HOTE** | Two stacked Taylor-KAN layers + LN + GELU + global avg pool | O(KDₕ(Dₕ+D)·T) |
| **DSMF** | PLE shared/specific experts; Top-*Kɡ* noisy sparse gating per modality | O(NₘKɡD²) |
| **Total** | **11.82 M params** (95% ResNet-18 backbone); 3.09 ms latency | — |

**Information-Theoretic Guarantee (Theorem A.7):**
`I(G_ISA; X,M) ≥ I(G_std; X,M) + H(M) − I(G_std; M) − 1`
with linear scaling `ΔI ≥ T·Hb(1−ρ) − 1` at sparsity ratio ρ.

---

## Main Results (F1-Score %, mean ± std, seeds 42/123/456)

| Method | CDNOW Value | CDNOW Churn | TAFENG Risk | TAFENG Repurchase | RETAIL Value | RETAIL Churn | INSTACART Activity | INSTACART Churn | SALES Risk | SALES Seasonality |
|--------|:-----------:|:-----------:|:-----------:|:-----------------:|:------------:|:------------:|:------------------:|:---------------:|:----------:|:-----------------:|
| TabM | 92.12±0.88 | 90.91±0.14 | 90.60±0.51 | 69.65±0.96 | 88.37±0.49 | 65.08±1.36 | 94.26±0.17 | 53.57±0.71 | 97.73±0.00 | 95.44±0.98 |
| TabICL | 89.28±0.00 | 91.27±0.00 | 88.56±0.00 | 71.87±0.00 | _91.70_±0.00 | 68.54±0.00 | 92.90±0.01 | 56.94±0.00 | **98.85**±0.00 | 94.40±0.00 |
| TabPFN | 90.26±0.00 | _91.44_±0.00 | 86.64±0.00 | 71.76±0.00 | 90.28±0.00 | 68.09±0.00 | 94.10±0.00 | 52.73±0.00 | 97.73±0.00 | 94.31±0.00 |
| ModernTCN | 92.62±0.55 | 71.81±24.03 | _91.62_±0.40 | 62.14±16.40 | 36.31±10.04 | 69.48±1.24 | 95.98±0.22 | 44.00±16.98 | **98.85**±1.16 | _97.35_±0.92 |
| xLSTM | _93.18_±0.12 | 74.22±19.86 | 88.45±1.20 | 61.94±18.55 | 66.84±40.29 | 47.03±19.57 | 94.12±0.85 | 55.30±2.14 | 89.20±15.71 | 93.70±2.33 |
| Hydra | 87.05±0.29 | 91.34±0.03 | 80.16±0.65 | 69.13±0.42 | 87.06±0.88 | 64.84±1.48 | 87.34±0.24 | 56.34±0.31 | **98.85**±0.00 | 94.55±0.05 |
| MPTSNet | 80.43±0.85 | 91.05±0.22 | 83.92±3.31 | 72.73±0.45 | 90.62±0.51 | 68.46±1.15 | _96.51_±1.05 | 54.80±1.36 | 98.46±0.68 | 96.28±5.06 |
| TimeMoE | 86.69±0.49 | 86.69±0.49 | 76.80±2.73 | 47.01±14.16 | 90.28±0.00 | 69.01±0.40 | 95.44±0.26 | 57.47±1.37 | 98.11±1.28 | 95.99±1.68 |
| DSN | 87.78±0.86 | 91.40±0.16 | 87.44±0.61 | _73.85_±0.34 | 89.39±0.52 | _71.48_±0.79 | 96.16±0.14 | _62.08_±0.35 | 91.34±0.87 | 97.28±1.08 |
| SoftShape | 85.23±0.58 | 91.17±0.17 | 86.73±0.84 | 73.81±0.32 | 86.19±1.94 | 69.71±1.07 | 88.88±6.03 | 61.98±0.46 | 89.13±0.85 | 95.28±2.42 |
| GAF-CNN | 93.15±0.58 | 62.05±37.54 | 83.94±5.33 | 71.24±1.90 | 45.29±37.93 | 53.80±17.90 | 96.16±0.11 | 42.43±18.58 | 97.70±1.11 | 93.77±0.47 |
| TimeLLM | 88.64±0.13 | 72.40±22.85 | 84.32±1.55 | 70.88±1.83 | 89.81±0.79 | 45.45±14.59 | 91.45±0.92 | 50.40±14.82 | 97.25±1.84 | 91.35±2.01 |
| GPT4TS | 93.12±0.12 | 74.22±19.86 | 85.86±0.12 | 61.94±18.55 | 66.84±40.29 | 47.03±19.57 | 94.55±0.82 | 54.20±1.45 | **98.85**±15.71 | 93.70±2.33 |
| **HMSPAR** | **93.20**±0.69 | **91.57**±0.15 | **92.72**±0.96 | **73.90**±0.36 | **92.30**±1.09 | **71.70**±1.82 | **96.54**±0.04 | **62.11**±1.18 | **98.85**±0.00 | **97.37**±1.25 |

**Bold** = best, _italic_ = second-best.

---

## Datasets

TAFENG is included under `data/tafeng/`. All other datasets require separate download.

| Dataset | Source | Primary Task | Auxiliary Task | Training Script |
|---------|--------|-------------|----------------|----------------|
| **TAFENG** *(included)* | [Kaggle](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset) | Risk Classification | Repurchase Detection | `train_tafeng.py` |
| CDNOW | [brucehardie.com](http://www.brucehardie.com/datasets/) | Value Classification | Churn Prediction | `train_cdnow.py` |
| RETAIL | [UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Value Classification | Churn Prediction | `train_retail.py` |
| INSTACART | [Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) | Activity Classification | Churn Prediction | `train_instacart.py` |
| SALES\_WEEKLY | [Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data) | Risk Classification | Seasonality Detection | `train_sales_weekly.py` |
| Simulated Merchant | `data/data_generator.py` | Anomaly Detection | — | `train_hmspar.py` |

All datasets are split 70 / 15 / 15 (train / val / test). Positive class ratio ≈ 35%.

### Data Directory

```
data/
├── tafeng/                             # included
│   ├── ta_feng_all_months_merged.csv
│   └── repurchase_task/
│       ├── amount_series.npy
│       ├── trans_series.npy
│       ├── isa_gaf_images.npy
│       ├── text_embeddings.npy
│       └── labels.npy
├── cdnow/                              # download separately
├── instacart/
├── retail/
└── sales_weekly/
```

---

## Baselines (13)

| Category | Models |
|----------|--------|
| Temporal-centric | xLSTM, ModernTCN, Hydra, SoftShape, DSN, MPTSNet, TimeMoE |
| Tabular-oriented | TabM, TabICL, TabPFN |
| Transformation & Foundation | TimeLLM, GPT4TS, GAF-CNN |

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.8, PyTorch ≥ 2.0, CUDA GPU recommended.

---

## Quick Start

### TAFENG (included)

```bash
python train_tafeng.py --multi-seed
```

### Simulated Merchant Dataset

```bash
python data/data_generator.py
python train_hmspar.py --industry Industry-0 --multi-seed
python train_hmspar.py --industry Industry-1 --multi-seed
python train_hmspar.py --industry Industry-2 --multi-seed
python train_hmspar.py --industry Industry-3 --multi-seed
```

### Public Datasets

```bash
python train_cdnow.py        --multi-seed
python train_instacart.py    --multi-seed
python train_retail.py       --multi-seed
python train_sales_weekly.py --multi-seed
```

### Baselines

```bash
cd baselines

# Primary task
python xlstm_baseline.py  --dataset tafeng --multi-seed
python tabm_baseline.py   --dataset tafeng --multi-seed
python cnn_baseline.py    --dataset tafeng --multi-seed

# Auxiliary task (repurchase / churn / seasonality)
python xlstm_baseline.py  --dataset tafeng --task repurchase --multi-seed
python dsn_baseline.py    --dataset cdnow  --task churn      --multi-seed

# Merchant simulated data (generate first)
python dsn_baseline.py    --dataset merchant --industry Industry-0 --multi-seed
```

---

## Evaluation Protocol

Results are reported as **mean ± std** across seeds {42, 123, 456} on the held-out test set.  
Primary metric: **F1-Score**. Additional metrics: Accuracy, Precision, Recall, AUC-ROC, AUPRC.

---

## Keywords

Sparse Sequence Classification · Homologous Multimodal Fusion · Sparsity-aware Representation · Mixture-of-Experts · Personalized Learning

---


