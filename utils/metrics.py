import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def compute_binary_metrics(y_true, y_pred_proba, threshold=0.5):
    
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'support': len(y_true)
    }
    
    
    if len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
    else:
        metrics['auc'] = np.nan
        metrics['auprc'] = np.nan
    
    return metrics


def print_metrics(metrics, prefix=""):
    print(f"\n{prefix}Evaluation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics.get('auc', np.nan):.4f}")
    print(f"  AUPRC:     {metrics.get('auprc', np.nan):.4f}")
    print(f"  Support:   {metrics['support']}")

