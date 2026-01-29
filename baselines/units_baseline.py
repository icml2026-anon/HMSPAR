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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, print_dataset_info, print_split_info
import argparse
import math


class UniTSDataset(Dataset):
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


class PatchEmbedding(nn.Module):
    """Patch-based tokenization for time series following UniTS paper"""
    def __init__(self, d_model, patch_len, stride, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        B, seq_len, n_vars = x.shape
        
        
        if seq_len < self.patch_len:
            
            padding_len = self.patch_len - seq_len
            x = F.pad(x, (0, 0, 0, padding_len))
            seq_len = self.patch_len
        
        
        x = x.transpose(1, 2)  # [B, n_vars, seq_len]
        
        
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, n_vars, n_patches, patch_len = x.shape
        
        
        x = x.reshape(B * n_vars, n_patches, patch_len)
        
        
        x = self.value_embedding(x)
        x = self.dropout(x)
        
        
        x = x.reshape(B, n_vars, n_patches, -1)
        
        return x, n_vars


class MultiHeadAttention(nn.Module):
    """Multi-head attention for UniTS"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        B, seq_len, d_model = query.shape
        
        
        Q = self.q_proj(query).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        
        return self.out_proj(out)


class SequenceAttentionBlock(nn.Module):
    """Sequence attention block from UniTS"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        
        B, n_vars, seq_len, d_model = x.shape
        
        
        x_reshaped = x.view(B * n_vars, seq_len, d_model)
        
        
        attn_out = self.self_attn(x_reshaped, x_reshaped, x_reshaped, mask)
        x_reshaped = self.norm1(x_reshaped + self.dropout(attn_out))
        
        
        return x_reshaped.view(B, n_vars, seq_len, d_model)


class VariableAttentionBlock(nn.Module):
    """Variable attention block from UniTS"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        B, n_vars, seq_len, d_model = x.shape
        
        
        x_reshaped = x.transpose(1, 2).contiguous().view(B * seq_len, n_vars, d_model)
        
        
        attn_out = self.self_attn(x_reshaped, x_reshaped, x_reshaped)
        x_reshaped = self.norm1(x_reshaped + self.dropout(attn_out))
        
        
        return x_reshaped.view(B, seq_len, n_vars, d_model).transpose(1, 2)


class DynamicLinear(nn.Module):
    """Dynamic linear layer from UniTS for adaptive feature processing"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class MLPBlock(nn.Module):
    """MLP block with dynamic linear layers following UniTS"""
    def __init__(self, d_model, mlp_ratio=4, dropout=0.1):
        super().__init__()
        d_ff = int(d_model * mlp_ratio)
        self.fc1 = DynamicLinear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = DynamicLinear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        
        identity = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.norm(identity + x)


class UniTSBlock(nn.Module):
    """Basic UniTS transformer block combining sequence, variable attention and MLP"""
    def __init__(self, d_model, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.seq_attn = SequenceAttentionBlock(d_model, num_heads, dropout)
        self.var_attn = VariableAttentionBlock(d_model, num_heads, dropout)
        self.mlp = MLPBlock(d_model, mlp_ratio, dropout)
    
    def forward(self, x, mask=None):
        x = self.seq_attn(x, mask)
        x = self.var_attn(x)
        x = self.mlp(x)
        return x


class ClassificationHead(nn.Module):
    """Classification head following UniTS design with cross-attention"""
    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        
        self.cross_attn = MultiHeadAttention(d_model, num_heads=8, dropout=dropout)
        
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        
        B, n_vars, seq_len, d_model = x.shape
        
        
        x_pooled = x.mean(dim=(1, 2))  # [B, d_model]
        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        
        
        x_pooled = x_pooled.unsqueeze(1)  # [B, 1, d_model]
        cls_out = self.cross_attn(cls_tokens, x_pooled, x_pooled)  # [B, 1, d_model]
        
        
        cls_out = cls_out.squeeze(1)  # [B, d_model]
        return self.mlp(cls_out)


class UniTSModel(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=128, num_heads=8, num_layers=4, 
                 patch_len=8, stride=8, mlp_ratio=4, dropout=0.1, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.num_classes = num_classes
        
        
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        
        
        self.n_patches = max(1, (seq_len - patch_len) // stride + 1)
        
        
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, self.n_patches, d_model))
        
        
        self.blocks = nn.ModuleList([
            UniTSBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        
        self.cls_head = ClassificationHead(d_model, num_classes, dropout)
        
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights following UniTS paper"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        
        
        torch.nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x):
        
        B, seq_len, input_dim = x.shape
        
        
        x, n_vars = self.patch_embedding(x)
        
        
        if x.shape[2] != self.pos_embedding.shape[2]:
            
            pos_emb = F.interpolate(
                self.pos_embedding.permute(0, 1, 3, 2), 
                size=x.shape[2], 
                mode='linear', 
                align_corners=False
            ).permute(0, 1, 3, 2)
        else:
            pos_emb = self.pos_embedding
        
        x = x + pos_emb[:, :n_vars]
        
        
        for block in self.blocks:
            x = block(x)
        
        
        output = self.cls_head(x)
        
        return output.squeeze(-1) if output.shape[-1] == 1 else output


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
    }


def train_epoch(model, loader, criterion, optimizer, device, clip_grad_norm=1.0):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
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


def main(args, seed=None):
    print("=" * 80)
    print(" UniTS Baseline ".center(80, "="))
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
    
    
    train_dataset = UniTSDataset(X_train, y_train)
    val_dataset = UniTSDataset(X_val, y_val)
    test_dataset = UniTSDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training UniTS ".center(80, "="))
    print("=" * 80)
    
    
    if X_train.ndim == 2:
        input_dim = 1  # Merchant: single feature per timestep
        seq_len = X_train.shape[1]
    else:
        input_dim = X_train.shape[2]  # Multiple features per timestep
        seq_len = X_train.shape[1]
    
    
    patch_len = min(args.patch_len, seq_len)
    stride = patch_len  # Non-overlapping patches as in UniTS
    
    print(f"Model config: input_dim={input_dim}, seq_len={seq_len}")
    print(f"              d_model={args.d_model}, num_heads={args.num_heads}, num_layers={args.num_layers}")
    print(f"              patch_len={patch_len}, stride={stride}")
    
    model = UniTSModel(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        patch_len=patch_len,
        stride=stride,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_classes=1  # Binary classification
    ).to(device)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        y_val_pred = (y_val_proba > 0.5).astype(int)
        val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
        
        scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Val F1={val_f1:.4f}, LR={current_lr:.2e}")
        
        if patience_counter >= args.patience:
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
    results_df['model'] = 'UniTS'
    results_df['d_model'] = args.d_model
    results_df['num_layers'] = args.num_layers
    results_df['patch_len'] = patch_len
    
    suffix = ''
    results_path = Config.RESULTS_DIR / f"units_{result_name}{suffix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)
    
    return test_metrics

def run_units_single_seed(args, seed):
    """Run UniTS with a specific seed and return results"""
    
    return main(args, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UniTS Baseline for Time Series Classification')
    parser.add_argument('--dataset', type=str, default='merchant',
                       choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'],
                       help='Dataset to use')
    parser.add_argument('--industry', type=str, default=None,
                       choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
                       help='Industry for merchant dataset')
    parser.add_argument('--task', type=str, default=None,
                       choices=['churn', 'seasonality', 'repurchase'],
                       help='Task type for appendix validation')
    
    
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--patch-len', type=int, default=8, help='Patch length for tokenization')
    parser.add_argument('--mlp-ratio', type=int, default=4, help='MLP expansion ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')
        
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - UNITS {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_units_single_seed(args, seed)
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