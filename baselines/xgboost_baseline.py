import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import xgboost as xgb
from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, flatten_time_series, print_dataset_info, print_split_info
import argparse


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
    }


def main(args):
    print("=" * 80)
    print(" XGBoost Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(Config.SEED)
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    
    
    X = flatten_time_series(X)
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\n" + "=" * 80)
    print(" Training XGBoost ".center(80, "="))
    print("=" * 80)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=Config.SEED,
        eval_metric='auc',
        early_stopping_rounds=20,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"Best iteration: {model.best_iteration}")
    
    print("\n" + "=" * 80)
    print(" Evaluation ".center(80, "="))
    print("=" * 80)
    
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba > 0.5).astype(int)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)
    
    print("\nValidation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba > 0.5).astype(int)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
    
    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    
    if args.dataset == 'merchant':
        result_name = args.industry.lower().replace("-", "")
    else:
        result_name = args.dataset
    
    results_df = pd.DataFrame([test_metrics])
    results_df['dataset'] = info['name']
    results_df['model'] = 'XGBoost'
    suffix = ''
    results_path = Config.RESULTS_DIR / f"xgboost_{result_name}{suffix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)

def run_xgboost_single_seed(args, seed):
    """Run XGBoost with a specific seed and return results"""
    print("=" * 80)
    print(" XGBoost Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(seed)
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    
    
    X = flatten_time_series(X)
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    print("\nTraining XGBoost...")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print("\n" + "=" * 80)
    print(" Test Results ".center(80, "="))
    print("=" * 80)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print("=" * 80)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='merchant',
                       choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'],
                       help='Dataset to use')
    parser.add_argument('--industry', type=str, default=None,
                       choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
                       help='Industry for merchant dataset')
    parser.add_argument('--task', type=str, default=None,
                       choices=['churn', 'seasonality', 'repurchase'],
                       help='Task type for appendix validation (churn, seasonality, repurchase)')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    args = parser.parse_args()
    
    
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')
    
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - XGBOOST {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_xgboost_single_seed(args, seed)
            all_results.append(result)
            
        
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY (3 SEEDS)")
        print("="*80)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
        
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Seeds: {seeds}")
        print("\nTest Results (Mean ± Std):")
        print("-" * 40)
        
        for metric in metrics:
            values = [result[metric] for result in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            
            if metric == 'f1':
                print(f"**{metric.upper():>10}: {mean_val:.4f} ± {std_val:.4f}**")
            else:
                print(f"{metric.upper():>12}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\nIndividual Results:")
        print("-" * 40)
        for i, (seed, result) in enumerate(zip(seeds, all_results)):
            print(f"Seed {seed}: F1 = {result['f1']:.4f}")
    else:
        
        main(args)

def run_xgboost_single_seed(args, seed):
    """Run XGBoost with a specific seed and return results"""
    print("=" * 80)
    print(" XGBoost Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(seed)
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    
    
    X = flatten_time_series(X)
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    print("\nTraining XGBoost...")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print("\n" + "=" * 80)
    print(" Test Results ".center(80, "="))
    print("=" * 80)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print("=" * 80)
    
    
    if args.dataset == 'merchant':
        result_name = args.industry.lower().replace("-", "")
    else:
        result_name = args.dataset
    
    results = pd.DataFrame([{
        'Dataset': info['name'],
        'Model': 'XGBoost',
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'AUC-ROC': metrics['auc'],
        'AUPRC': metrics['auprc'],
    }])
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    suffix = ''
    result_filename = f'xgboost_{result_name}{suffix}_results.csv'
    result_path = Config.RESULTS_DIR / result_filename
    results.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")
    
    return metrics

