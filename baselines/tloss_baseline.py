#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, print_dataset_info, print_split_info


class TemporalLoss(nn.Module):
    """Temporal Loss for encouraging temporal consistency"""
    
    def __init__(self, alpha=1.0, beta=1.0):
        super(TemporalLoss, self).__init__()
        self.alpha = alpha  # Weight for temporal smoothness
        self.beta = beta    # Weight for temporal variation
        
    def forward(self, features, labels=None):
        """
        Args:
            features: [B, seq_len, feature_dim] temporal features
            labels: [B] labels (optional, for supervised temporal loss)
        """
        batch_size, seq_len, feature_dim = features.shape
        
        
        if seq_len > 1:
            diff = features[:, 1:, :] - features[:, :-1, :]  # [B, seq_len-1, feature_dim]
            smoothness_loss = torch.mean(torch.norm(diff, dim=-1)**2)
        else:
            smoothness_loss = torch.tensor(0.0, device=features.device)
        
        
        temporal_mean = torch.mean(features, dim=1, keepdim=True)  # [B, 1, feature_dim]
        variation = features - temporal_mean  # [B, seq_len, feature_dim]
        variation_loss = -torch.mean(torch.norm(variation, dim=-1)**2)  # Negative to encourage variation
        
        total_loss = self.alpha * smoothness_loss + self.beta * variation_loss
        
        return total_loss, {
            'smoothness_loss': smoothness_loss.item(),
            'variation_loss': variation_loss.item()
        }


class TLossNet(nn.Module):
    """Neural Network with Temporal Loss for time series classification"""
    
    def __init__(self, input_dim=1, seq_len=50, hidden_dim=128, num_layers=2, 
                 num_classes=2, dropout=0.3, use_lstm=True):
        super(TLossNet, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
        
        
        if use_lstm:
            self.temporal_encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            feature_dim = hidden_dim * 2  # Bidirectional
        else:
            
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
                nn.ReLU()
            )
            feature_dim = hidden_dim * 2
        
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        
        self.temporal_loss = TemporalLoss(alpha=1.0, beta=0.1)
        
    def forward(self, x, return_features=False):
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, seq_len, 1] for univariate
            
        batch_size, seq_len, input_dim = x.shape
        
        
        if self.use_lstm:
            features, _ = self.temporal_encoder(x)  # [B, seq_len, hidden_dim*2]
        else:
            
            x_cnn = x.transpose(1, 2)
            features = self.temporal_encoder(x_cnn)  # [B, hidden_dim*2, seq_len]
            features = features.transpose(1, 2)  # [B, seq_len, hidden_dim*2]
        
        
        attention_weights = self.attention(features)  # [B, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # [B, seq_len, 1]
        
        
        aggregated_features = torch.sum(features * attention_weights, dim=1)  # [B, hidden_dim*2]
        
        
        logits = self.classifier(aggregated_features)
        
        if return_features:
            return logits, features, attention_weights
        else:
            return logits, features


class AdvancedTLossNet(nn.Module):
    """Advanced T-Loss Network with multi-scale temporal modeling"""
    
    def __init__(self, input_dim=1, seq_len=50, hidden_dim=128, num_classes=2, dropout=0.3):
        super(AdvancedTLossNet, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        
        self.short_term_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.long_term_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=4, padding=4),
        ])
        
        
        total_feature_dim = hidden_dim * 2 * 2 + hidden_dim * len(self.dilated_convs)  # LSTM outputs + conv outputs
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        
        self.temporal_loss_short = TemporalLoss(alpha=1.0, beta=0.1)
        self.temporal_loss_long = TemporalLoss(alpha=0.5, beta=0.2)
        
    def forward(self, x):
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, seq_len, 1] for univariate
            
        batch_size, seq_len, input_dim = x.shape
        
        
        short_features, _ = self.short_term_encoder(x)  # [B, seq_len, hidden_dim*2]
        long_features, _ = self.long_term_encoder(x)    # [B, seq_len, hidden_dim*2]
        
        
        x_conv = x.transpose(1, 2)  # [B, input_dim, seq_len]
        conv_features = []
        for conv in self.dilated_convs:
            conv_out = F.relu(conv(x_conv))  # [B, hidden_dim, seq_len]
            conv_out = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)  # [B, hidden_dim]
            conv_features.append(conv_out)
        
        
        short_pooled = torch.mean(short_features, dim=1)  # [B, hidden_dim*2]
        long_pooled = torch.mean(long_features, dim=1)    # [B, hidden_dim*2]
        
        
        all_features = torch.cat([short_pooled, long_pooled] + conv_features, dim=1)
        
        
        fused_features = self.feature_fusion(all_features)
        
        
        logits = self.classifier(fused_features)
        
        return logits, short_features, long_features


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
    }


def train_epoch(model, loader, criterion, optimizer, device, tloss_weight=0.1):
    model.train()
    total_loss = 0
    total_tloss = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        if isinstance(model, AdvancedTLossNet):
            logits, short_features, long_features = model(X_batch)
            
            
            cls_loss = criterion(logits, y_batch.long())
            
            
            t_loss_short, _ = model.temporal_loss_short(short_features)
            t_loss_long, _ = model.temporal_loss_long(long_features)
            t_loss = t_loss_short + t_loss_long
        else:
            logits, features = model(X_batch)
            
            
            cls_loss = criterion(logits, y_batch.long())
            
            
            t_loss, _ = model.temporal_loss(features)
        
        
        loss = cls_loss + tloss_weight * t_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += cls_loss.item()
        total_tloss += t_loss.item()
    
    return total_loss / len(loader), total_tloss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            
            if isinstance(model, AdvancedTLossNet):
                logits, _, _ = model(X_batch)
            else:
                logits, _ = model(X_batch)
            
            
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_labels), np.array(all_probs)


def main(args, seed=None):
    print("=" * 80)
    print(" T-Loss Baseline ".center(80, "="))
    print("=" * 80)
    
    if seed is not None:
        set_seed(seed)
    else:
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
    print(f"T-loss weight: {args.tloss_weight}")
    
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training T-Loss Model ".center(80, "="))
    print("=" * 80)
    
    
    if args.model_type == 'advanced':
        model = AdvancedTLossNet(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=args.hidden_dim,
            num_classes=2,
            dropout=args.dropout
        ).to(device)
    else:
        model = TLossNet(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout,
            use_lstm=args.use_lstm
        ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        
        train_loss, train_tloss = train_epoch(model, train_loader, criterion, optimizer, device, args.tloss_weight)
        
        
        val_labels, val_probs = evaluate(model, val_loader, device)
        val_preds = (val_probs > 0.5).astype(int)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        
        scheduler.step(val_metrics['f1'])
        
        print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, T-Loss={train_tloss:.4f}, Val F1={val_metrics['f1']:.4f}")
        
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_tloss_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 80)
    print(" Evaluation ".center(80, "="))
    print("=" * 80)
    
    
    if os.path.exists('best_tloss_model.pth'):
        model.load_state_dict(torch.load('best_tloss_model.pth'))
    test_labels, test_probs = evaluate(model, test_loader, device)
    test_preds = (test_probs > 0.5).astype(int)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    
    results_dir = '/root/HMSPAR/results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'tloss_{args.dataset}_results.csv')
    results_df = pd.DataFrame([test_metrics])
    results_df['model'] = f'T-Loss_{args.model_type}'
    results_df['dataset'] = args.dataset
    results_df['tloss_weight'] = args.tloss_weight
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    
    if os.path.exists('best_tloss_model.pth'):
        os.remove('best_tloss_model.pth')
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)
    
    return test_metrics

def run_tloss_single_seed(args, seed):
    """Run TLoss with a specific seed and return results"""
    
    return main(args, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T-Loss Baseline for Time Series Classification')
    
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'],
                        help='Dataset to use')
    parser.add_argument('--industry', type=str, 
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
                        help='Industry for merchant dataset')
    parser.add_argument('--task', type=str, choices=['engagement', 'churn', 'seasonality', 'repurchase', 'sales_risk_classification', 'customer_risk_classification'],
                        help='Specific task for multi-task datasets')
    
    
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'advanced'],
                        help='Type of T-Loss model to use')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--use-lstm', action='store_true', default=True,
                        help='Use LSTM instead of CNN for temporal encoding')
    parser.add_argument('--tloss-weight', type=float, default=0.1,
                        help='Weight for temporal loss')
    
    
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
        print(f"MULTI-SEED STATISTICAL ANALYSIS - TLOSS {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_tloss_single_seed(args, seed)
            all_results.append(result)
            
        
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY (3 SEEDS)")
        print("="*80)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
        
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Seeds: {seeds}")
        print("\nTest Results (Mean ± Std):")
        print("-" * 40)
        
        import numpy as np
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
