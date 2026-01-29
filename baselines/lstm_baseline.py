import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, print_dataset_info, print_split_info
import argparse


class LSTMDataset(Dataset):
    def __init__(self, X, y):
        
        if X.ndim == 2:
            
            self.X = torch.FloatTensor(X).unsqueeze(-1)
        else:
            self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze()


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return np.array(all_labels), np.array(all_probs)


def main(args):
    print("=" * 80)
    print(" LSTM Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    print(f"Device: {device}")
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    train_dataset = LSTMDataset(X_train, y_train)
    val_dataset = LSTMDataset(X_val, y_val)
    test_dataset = LSTMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training LSTM ".center(80, "="))
    print("=" * 80)
    
    
    if X_train.ndim == 2:
        input_dim = 1  # Merchant: single feature per timestep
    else:
        input_dim = X_train.shape[2]  # Multiple features per timestep
    
    model = LSTMClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    
    max_epochs = 10 if args.dataset == 'instacart' else 50
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        y_val_pred = (y_val_proba > 0.5).astype(int)
        val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val F1={val_f1:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model)
    
    print("\n" + "=" * 80)
    print(" Evaluation ".center(80, "="))
    print("=" * 80)
    
    y_test_true, y_test_proba = evaluate(model, test_loader, device)
    y_test_pred = (y_test_proba > 0.5).astype(int)
    test_metrics = compute_metrics(y_test_true, y_test_pred, y_test_proba)
    
    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    
    if args.dataset == 'merchant':
        result_name = args.industry.lower().replace("-", "")
    else:
        result_name = args.dataset
    
    results_df = pd.DataFrame([test_metrics])
    results_df['dataset'] = info['name']
    results_df['model'] = 'LSTM'
    
    suffix = ''
    results_path = Config.RESULTS_DIR / f"lstm_{result_name}{suffix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)

def run_lstm_single_seed(args, seed):
    """Run LSTM with a specific seed and return results"""
    print("=" * 80)
    print(" LSTM Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    print(f"Device: {device}")
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    train_dataset = LSTMDataset(X_train, y_train)
    val_dataset = LSTMDataset(X_val, y_val)
    test_dataset = LSTMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training LSTM ".center(80, "="))
    print("=" * 80)
    
    
    if X_train.ndim == 2:
        input_dim = 1  # Single feature per timestep
    else:
        input_dim = X_train.shape[2]  # Multiple features per timestep
    
    model = LSTMClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    
    max_epochs = 10 if args.dataset == 'instacart' else 50
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        y_val_pred = (y_val_proba > 0.5).astype(int)
        val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
        
        print(f"Epoch {epoch+1:2d}/{max_epochs} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    y_test_true, y_test_proba = evaluate(model, test_loader, device)
    y_test_pred = (y_test_proba > 0.5).astype(int)
    
    test_metrics = compute_metrics(y_test_true, y_test_pred, y_test_proba)
    
    print("\n" + "=" * 80)
    print(" Test Results ".center(80, "="))
    print("=" * 80)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    
    if os.path.exists('best_lstm_model.pth'):
        os.remove('best_lstm_model.pth')
    
    return test_metrics


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
        print(f"MULTI-SEED STATISTICAL ANALYSIS - LSTM {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_lstm_single_seed(args, seed)
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

def run_lstm_single_seed(args, seed):
    """Run LSTM with a specific seed and return results"""
    print("=" * 80)
    print(" LSTM Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    print(f"Device: {device}")
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    train_dataset = LSTMDataset(X_train, y_train)
    val_dataset = LSTMDataset(X_val, y_val)
    test_dataset = LSTMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training LSTM ".center(80, "="))
    print("=" * 80)
    
    
    if X_train.ndim == 2:
        input_dim = 1  # Single feature per timestep
    else:
        input_dim = X_train.shape[2]  # Multiple features per timestep
    
    model = LSTMClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    
    max_epochs = 10 if args.dataset == 'instacart' else 50
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        y_val_pred = (y_val_proba > 0.5).astype(int)
        val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
        
        print(f"Epoch {epoch+1:2d}/{max_epochs} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    y_test_true, y_test_proba = evaluate(model, test_loader, device)
    y_test_pred = (y_test_proba > 0.5).astype(int)
    
    test_metrics = compute_metrics(y_test_true, y_test_pred, y_test_proba)
    
    print("\n" + "=" * 80)
    print(" Test Results ".center(80, "="))
    print("=" * 80)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    
    if os.path.exists('best_lstm_model.pth'):
        os.remove('best_lstm_model.pth')
    
    return test_metrics

