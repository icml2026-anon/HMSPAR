import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from configs.config import Config
from data_utils import load_dataset, prepare_data_splits, flatten_time_series, print_dataset_info, print_split_info


def main():
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
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--min_samples_split', type=int, default=10)
    parser.add_argument('--min_samples_leaf', type=int, default=5)
    parser.add_argument('--max_features', type=str, default='sqrt')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    args = parser.parse_args()
    
    
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')
        
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - RANDOMFOREST {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_randomforest_single_seed(args, seed)
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
            
        return
        
    
    run_randomforest_single_seed(args, 42)

def run_randomforest_single_seed(args, seed):
    """Run Random Forest with a specific seed and return results"""
    
    print("=" * 80)
    print(" Random Forest Baseline ".center(80, "="))
    print("=" * 80)
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    
    
    X = flatten_time_series(X)
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    print("\nTraining Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=seed,
        n_jobs=args.n_jobs,
        verbose=1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.0
    
    try:
        auprc = average_precision_score(y_test, y_proba)
    except:
        auprc = 0.0
    
    print("\n" + "=" * 80)
    print(" Test Results ".center(80, "="))
    print("=" * 80)
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    print("=" * 80)
    
    
    if args.dataset == 'merchant':
        result_name = args.industry.lower().replace("-", "")
    else:
        result_name = args.dataset
    
    results = pd.DataFrame([{
        'Dataset': info['name'],
        'Model': 'RandomForest',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'AUPRC': auprc,
    }])
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    suffix = ''
    result_filename = f'randomforest_{result_name}{suffix}_results.csv'
    result_path = Config.RESULTS_DIR / result_filename
    results.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")
    
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'auprc': auprc
    }


if __name__ == '__main__':
    main()

