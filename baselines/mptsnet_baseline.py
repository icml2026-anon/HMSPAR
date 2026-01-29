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


class MPTSNetDataset(Dataset):
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


def fft_analysis(data):
    """FFT analysis to find dominant periods"""
    
    batch_size, seq_length, num_channels = data.shape
    
    
    averaged_data = data.mean(dim=-1)  # [B, T]
    
    
    yf = torch.fft.fft(averaged_data, dim=1)
    power_spectrum = torch.abs(yf)[:, :seq_length//2]  # [B, T//2]
    
    
    freqs = torch.fft.fftfreq(seq_length)[:seq_length//2]
    
    
    valid_mask = freqs > 1/seq_length  # Avoid very low frequencies
    if valid_mask.sum() == 0:
        
        return [2, 3, 4, 5]
    
    valid_freqs = freqs[valid_mask]
    valid_power = power_spectrum[:, valid_mask].mean(dim=0)  # Average across batch
    
    
    top_indices = torch.argsort(valid_power, descending=True)[:4]
    periods = []
    
    for idx in top_indices:
        freq = valid_freqs[idx]
        period = int(round(1.0 / freq.item()))
        if 2 <= period <= seq_length // 2:  # Valid period range
            periods.append(period)
    
    
    while len(periods) < 4:
        periods.append(min(2 + len(periods), seq_length // 2))
    
    return periods[:4]


def fft_find_amplitude(data, target_period):
    """Find amplitude for a specific period in batch data"""
    batch_size, seq_length, num_channels = data.shape
    
    
    averaged_data = data.mean(dim=-1)  # [B, T]
    
    
    yf = torch.fft.fft(averaged_data, dim=1)
    power_spectrum = torch.abs(yf)[:, :seq_length//2]
    
    
    target_freq = seq_length / target_period
    freqs = torch.arange(seq_length//2).float()
    
    
    freq_diff = torch.abs(freqs - target_freq)
    closest_idx = torch.argmin(freq_diff)
    
    
    amplitudes = power_spectrum[:, closest_idx].unsqueeze(1)  # [B, 1]
    
    return amplitudes


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        return x + self.pe[:, :x.size(1), :x.size(2)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, embed_dim):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=embed_dim, 
                                  kernel_size=3, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, embed_dim, seq_length, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, embed_dim=embed_dim)
        self.position_embedding = PositionalEncoding(embed_dim=embed_dim, max_len=seq_length)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        x = self.value_embedding(x)  # [B, T, embed_dim]
        x = x + self.position_embedding(x)  # Add positional encoding
        return self.dropout(x)


class InceptionBlock(nn.Module):
    """Simplified Inception block for local pattern extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        
        self.conv1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        xp = self.maxpool(x)
        
        out = torch.cat([x1, x3, x5, xp], dim=1)
        out = self.bn(out)
        return self.activation(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        
        attn_out, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_out))
        
        
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_out))
        
        return x


class PeriodicBlock(nn.Module):
    """Core periodic processing block of MPTSNet"""
    def __init__(self, periods, seq_length, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.periods = periods
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        
        self.inception = InceptionBlock(embed_dim, embed_dim)
        
        
        self.global_transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        
        self.period_transformers = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, ff_dim)
                for _ in range(num_layers)
            ])
            for _ in periods
        ])
        
    def forward(self, x):
        
        B, C, T = x.shape
        
        
        x_global = x.permute(0, 2, 1)  # [B, T, embed_dim]
        for layer in self.global_transformer:
            x_global = layer(x_global)
        time_features = x_global.permute(0, 2, 1)  # [B, embed_dim, T]
        
        
        period_features = []
        amplitudes = []
        
        for i, period in enumerate(self.periods):
            
            x_for_amp = x.permute(0, 2, 1)  # [B, T, embed_dim] 
            amp = fft_find_amplitude(x_for_amp, period).to(x.device)  # [B, 1]
            amplitudes.append(amp)
            
            
            if T % period != 0:
                
                pad_len = ((T // period) + 1) * period - T
                x_pad = F.pad(x, (0, pad_len))
                T_pad = x_pad.shape[2]
            else:
                x_pad = x
                T_pad = T
                
            num_windows = T_pad // period
            
            
            x_windows = x_pad.view(B, C, period, num_windows)
            
            
            window_features = []
            for j in range(num_windows):
                window = x_windows[:, :, :, j]  # [B, embed_dim, period]
                local_feat = self.inception(window)  # [B, embed_dim, period]
                window_features.append(local_feat)
            
            
            local_features = torch.stack(window_features, dim=-1)  # [B, embed_dim, period, num_windows]
            local_features = local_features + x_windows
            
            
            local_features = local_features.permute(0, 3, 1, 2).contiguous()  # [B, num_windows, embed_dim, period]
            
            local_features = local_features.mean(dim=-1)  # [B, num_windows, embed_dim]
            
            
            for layer in self.period_transformers[i]:
                local_features = layer(local_features)
            
            
            
            local_features = local_features.permute(0, 2, 1).contiguous()  # [B, embed_dim, num_windows]
            
            
            global_feat = F.interpolate(local_features, size=T, mode='linear', align_corners=False)
            
            
            global_feat = global_feat[:, :, :T]
            period_features.append(global_feat)
        
        
        if len(amplitudes) > 0:
            amplitudes = torch.cat(amplitudes, dim=1)  # [B, num_periods]
            weights = F.softmax(amplitudes, dim=1)  # [B, num_periods]
            
            
            period_features = torch.stack(period_features, dim=-1)
            
            
            period_weights = weights.view(B, 1, 1, -1)
            weighted_features = (period_features * period_weights).sum(dim=-1)  # [B, embed_dim, T]
        else:
            weighted_features = torch.zeros_like(time_features)
        
        
        output = time_features + weighted_features + x
        
        return output


class MPTSNetModel(nn.Module):
    def __init__(self, input_dim, seq_len, embed_dim=128, num_heads=8, ff_dim=256, 
                 num_layers=2, num_blocks=2, num_classes=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        
        self.embedding = DataEmbedding(input_dim, embed_dim, seq_len)
        
        
        self.periods = [2, 3, 4, 5]  # Default periods
        
        
        self.periodic_blocks = nn.ModuleList([
            PeriodicBlock(
                periods=self.periods,
                seq_length=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers
            )
            for _ in range(num_blocks)
        ])
        
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(seq_len * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        B = x.shape[0]
        
        
        if self.training:
            try:
                detected_periods = fft_analysis(x)
                self.periods = detected_periods
                
                for block in self.periodic_blocks:
                    block.periods = self.periods
            except:
                
                pass
        
        
        x = self.embedding(x)
        
        
        x = x.permute(0, 2, 1)
        
        
        for block in self.periodic_blocks:
            x = self.layer_norm(block(x).permute(0, 2, 1)).permute(0, 2, 1)
        
        
        x = x.permute(0, 2, 1)  # [B, T, embed_dim]
        x = x.reshape(B, -1)  # [B, T * embed_dim]
        
        
        logits = self.classifier(x)
        
        
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        
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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_labels), np.array(all_probs)


def main(args, seed=None):
    print("=" * 80)
    print(" MPTSNet Baseline ".center(80, "="))
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
    
    
    
    train_dataset = MPTSNetDataset(X_train, y_train)
    val_dataset = MPTSNetDataset(X_val, y_val)
    test_dataset = MPTSNetDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training MPTSNet ".center(80, "="))
    print("=" * 80)
    
    
    if X_train.ndim == 2:
        input_dim = 1
        seq_len = X_train.shape[1]
    else:
        input_dim = X_train.shape[2]
        seq_len = X_train.shape[1]
    
    print(f"Model config: input_dim={input_dim}, seq_len={seq_len}")
    print(f"              embed_dim={args.embed_dim}, num_heads={args.num_heads}")
    print(f"              ff_dim={args.ff_dim}, num_layers={args.num_layers}, num_blocks={args.num_blocks}")
    
    model = MPTSNetModel(
        input_dim=input_dim,
        seq_len=seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        num_blocks=args.num_blocks,
        num_classes=1  # Binary classification
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_f1 = 0
    patience_counter = 0
    best_model = model.state_dict()
    
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
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Val F1={val_f1:.4f}")
        
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
    results_df['model'] = 'MPTSNet'
    results_df['embed_dim'] = args.embed_dim
    results_df['num_blocks'] = args.num_blocks
    
    suffix = ''
    results_path = Config.RESULTS_DIR / f"mptsnet_{result_name}{suffix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MPTSNet Baseline for Time Series Classification')
    parser.add_argument('--dataset', type=str, default='merchant',
                       choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                       choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                       choices=['churn', 'seasonality', 'repurchase'])
    
    
    parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=256, help='Feed-forward dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of transformer layers per block')
    parser.add_argument('--num-blocks', type=int, default=2, help='Number of periodic blocks')
    
    
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')

def run_mptsnet_single_seed(args, seed):
    """Run MPTSNet with a specific seed and return results"""
    
    return main(args, seed)


if args.multi_seed:
    seeds = [42, 123, 456]
    all_results = []
    
    print("="*80)
    print(f"MULTI-SEED STATISTICAL ANALYSIS - MPTSNET {args.dataset.upper()}")
    print("="*80)
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[SEED {i}/3] Running with seed = {seed}")
        print("-" * 50)
        
        result = run_mptsnet_single_seed(args, seed)
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
