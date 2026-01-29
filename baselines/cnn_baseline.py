#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, print_dataset_info, print_split_info


class CNN1D(nn.Module):
    """1D CNN for time series classification"""
    
    def __init__(self, input_dim=1, seq_len=50, num_classes=2, 
                 num_filters=[64, 128, 256], kernel_sizes=[3, 5, 7],
                 dropout=0.3):
        super(CNN1D, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        
        in_channels = input_dim
        for i, (num_filter, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            self.conv_layers.append(
                nn.Conv1d(in_channels, num_filter, kernel_size, padding=kernel_size//2)
            )
            self.bn_layers.append(nn.BatchNorm1d(num_filter))
            in_channels = num_filter
        
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        
        
        feature_size = 2 * num_filters[-1]
        
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, seq_len, 1] for univariate
            
        
        x = x.transpose(1, 2)
        
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            
            if x.size(-1) >= 2:
                x = F.max_pool1d(x, kernel_size=2, stride=1, padding=0)
            
        
        
        max_pooled = self.global_max_pool(x).squeeze(-1)  # [B, num_filters[-1]]
        avg_pooled = self.global_avg_pool(x).squeeze(-1)  # [B, num_filters[-1]]
        
        
        features = torch.cat([max_pooled, avg_pooled], dim=1)  # [B, 2*num_filters[-1]]
        
        
        logits = self.classifier(features)
        return logits




def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
    }


def train_epoch(model, loader, criterion, optimizer, device, epoch=0, dataset_name=""):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN/inf detected in logits, skipping batch")
            continue
            
        loss = criterion(logits, y_batch.long())
        
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/inf detected in loss, skipping batch")
            continue
        
        loss.backward()
        
        
        if dataset_name.lower() == "instacart":
            
            if epoch < 3:
                max_grad_norm = 1.0
                skip_threshold = 500.0
            elif epoch < 10:
                max_grad_norm = 0.5  
                skip_threshold = 200.0
            else:
                max_grad_norm = 0.3
                skip_threshold = 100.0
        else:
            
            if epoch < 3:
                max_grad_norm = 5.0
                skip_threshold = 200.0
            elif epoch < 10:
                max_grad_norm = 2.0  
                skip_threshold = 100.0
            else:
                max_grad_norm = 1.0
                skip_threshold = 75.0
                
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        
        if grad_norm > skip_threshold:
            print(f"Warning: Extreme gradient norm {grad_norm:.4f}, skipping update")
            continue
            
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
            logits = model(X_batch)
            
            
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_labels), np.array(all_probs)


def main(args):
    print("=" * 80)
    print(" CNN Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    print(f"Device: {device}")
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    if X_train.ndim == 2:
        seq_len, input_dim = X_train.shape[1], 1
        X_train = X_train.reshape(X_train.shape[0], seq_len, 1)
        X_val = X_val.reshape(X_val.shape[0], seq_len, 1)
        X_test = X_test.reshape(X_test.shape[0], seq_len, 1)
    else:
        seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    
    print(f"\nModel config: input_dim={input_dim}, seq_len={seq_len}")
    
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training CNN ".center(80, "="))
    print("=" * 80)
    
    
    model = CNN1D(
        input_dim=input_dim,
        seq_len=seq_len,
        num_classes=2,
        num_filters=[64, 128, 256],
        kernel_sizes=[3, 5, 7],
        dropout=args.dropout
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    
    if args.dataset.lower() == "instacart":
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr * 0.01,  # Reduce learning rate by 100x for Instacart
            weight_decay=args.weight_decay * 20,  # Heavy regularization
            eps=1e-8,
            betas=(0.9, 0.999)
        )
    else:
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr * 0.1,  # Reduce learning rate by 10x
            weight_decay=args.weight_decay * 10,  # Increase regularization
            eps=1e-8,
            betas=(0.9, 0.999)
        )
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.2
    )
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.dataset)
        
        
        val_labels, val_probs = evaluate(model, val_loader, device)
        val_preds = (val_probs > 0.5).astype(int)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        
        scheduler.step(val_metrics['f1'])
        
        print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Val F1={val_metrics['f1']:.4f}")
        
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 80)
    print(" Evaluation ".center(80, "="))
    print("=" * 80)
    
    
    if os.path.exists('best_cnn_model.pth'):
        model.load_state_dict(torch.load('best_cnn_model.pth'))
    else:
        print("Warning: No saved model found, using current model for evaluation")
    test_labels, test_probs = evaluate(model, test_loader, device)
    test_preds = (test_probs > 0.5).astype(int)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    
    results_dir = '/root/HMSPAR/results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'cnn_{args.dataset}_results.csv')
    results_df = pd.DataFrame([test_metrics])
    results_df['model'] = 'CNN'
    results_df['dataset'] = args.dataset
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    
    if os.path.exists('best_cnn_model.pth'):
        os.remove('best_cnn_model.pth')
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)

def run_cnn_single_seed(args, seed):
    """Run CNN with a specific seed and return results"""
    print("=" * 80)
    print(" CNN Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    print(f"Device: {device}")
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    if X_train.ndim == 2:
        seq_len, input_dim = X_train.shape[1], 1
        X_train = X_train.reshape(X_train.shape[0], seq_len, 1)
        X_val = X_val.reshape(X_val.shape[0], seq_len, 1)
        X_test = X_test.reshape(X_test.shape[0], seq_len, 1)
    else:
        seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"\nModel config: input_dim={input_dim}, seq_len={seq_len}")
    
    print("\n" + "=" * 80)
    print(" Training CNN ".center(80, "="))
    print("=" * 80)
    
    model = CNN1D(
        input_dim=input_dim,
        seq_len=seq_len,
        num_classes=2,
        num_filters=[64, 128, 256],
        kernel_sizes=[3, 5, 7],
        dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_f1 = -1
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.dataset)
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        y_val_pred = (y_val_proba > 0.5).astype(int)
        val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    
    model.load_state_dict(torch.load('best_cnn_model.pth'))
    y_test_true, y_test_proba = evaluate(model, test_loader, device)
    y_test_pred = (y_test_proba > 0.5).astype(int)
    
    test_metrics = compute_metrics(y_test_true, y_test_pred, y_test_proba)
    
    print("\n" + "=" * 80)
    print(" Test Results ".center(80, "="))
    print("=" * 80)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    
    if os.path.exists('best_cnn_model.pth'):
        os.remove('best_cnn_model.pth')
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Baseline for Time Series Classification')
    
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'],
                        help='Dataset to use')
    parser.add_argument('--industry', type=str, 
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
                        help='Industry for merchant dataset')
    parser.add_argument('--task', type=str, choices=['churn', 'seasonality', 'repurchase'],
                        help='Specific task for multi-task datasets')
    
    
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - CNN {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_cnn_single_seed(args, seed)
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

def run_cnn_single_seed(args, seed):
    """Run CNN with a specific seed and return results"""
    print("=" * 80)
    print(" CNN Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    print(f"Device: {device}")
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    if X_train.ndim == 2:
        seq_len, input_dim = X_train.shape[1], 1
        X_train = X_train.reshape(X_train.shape[0], seq_len, 1)
        X_val = X_val.reshape(X_val.shape[0], seq_len, 1)
        X_test = X_test.reshape(X_test.shape[0], seq_len, 1)
    else:
        seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    
    print(f"\nModel config: input_dim={input_dim}, seq_len={seq_len}")
    
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training CNN ".center(80, "="))
    print("=" * 80)
    
    
    model = CNN1D(input_dim=input_dim, seq_len=seq_len, dropout=args.dropout).to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    best_val_f1 = -1
    patience_counter = 0
    
    for epoch in range(args.epochs):
        
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        
        val_labels, val_probs = evaluate(model, val_loader, device)
        val_preds = (val_probs > 0.5).astype(int)
        
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_f1 = val_metrics['f1']
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    
    if os.path.exists('best_cnn_model.pth'):
        model.load_state_dict(torch.load('best_cnn_model.pth'))
    else:
        print("Warning: No saved model found, using current model for evaluation")
    test_labels, test_probs = evaluate(model, test_loader, device)
    test_preds = (test_probs > 0.5).astype(int)
    
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    print("\n" + "=" * 80)
    print(" Test Results ".center(80, "="))
    print("=" * 80)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    
    if os.path.exists('best_cnn_model.pth'):
        os.remove('best_cnn_model.pth')
    
    return test_metrics
