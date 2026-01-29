import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from configs.config import Config
from data_utils import load_dataset, prepare_data_splits, flatten_time_series, print_dataset_info, print_split_info


class MLP_Block(nn.Module):
    def __init__(self, d_in: int, d: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d, d_in)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModernNCA(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dim: int = 128,
        dropout: float = 0.1,
        d_block: int = 512,
        n_blocks: int = 0,
        temperature: float = 1.0,
        sample_rate: float = 0.5
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dim = dim
        self.dropout = dropout
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.T = temperature
        self.sample_rate = sample_rate
        
        if n_blocks > 0:
            self.post_encoder = nn.Sequential(*[
                MLP_Block(dim, d_block, dropout)
                for _ in range(n_blocks)
            ], nn.BatchNorm1d(dim))
        self.encoder = nn.Linear(self.d_in, dim)

    def forward(self, x, y, candidate_x, candidate_y, is_train):
        if is_train:
            data_size = candidate_x.shape[0]
            retrieval_size = int(data_size * self.sample_rate)
            sample_idx = torch.randperm(data_size, device=x.device)[:retrieval_size]
            candidate_x = candidate_x[sample_idx]
            candidate_y = candidate_y[sample_idx]
        
        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)
        
        if self.n_blocks > 0:
            x = self.post_encoder(x)
            candidate_x = self.post_encoder(candidate_x)
        
        if is_train:
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])
        
        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y = candidate_y.unsqueeze(-1)
        
        distances = torch.cdist(x, candidate_x, p=2)
        distances = distances / self.T
        
        if is_train:
            distances = distances.fill_diagonal_(torch.inf)
        
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        
        if self.d_out > 1:
            eps = 1e-7
            logits = torch.log(logits + eps)
        
        return logits.squeeze(-1)


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, train_loader, candidate_x, candidate_y, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x, batch_y, candidate_x, candidate_y, is_train=True)
        loss = F.nll_loss(logits, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, candidate_x, candidate_y, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x, None, candidate_x, candidate_y, is_train=False)
            probs = torch.exp(logits)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    try:
        auprc = average_precision_score(all_labels, all_probs)
    except:
        auprc = 0.0
    
    return accuracy, precision, recall, f1, auc, auprc


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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--d_block', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--sample_rate', type=float, default=0.7)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    args = parser.parse_args()
    
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')
        
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - MODERNNCA {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_modernnca_single_seed(args, seed)
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
        
    
    run_modernnca_single_seed(args, 42)

def run_modernnca_single_seed(args, seed):
    """Run ModernNCA with a specific seed and return results"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    
    
    X = flatten_time_series(X)
    
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    
    n_candidates = max(100, int(0.1 * len(X_train)))
    candidate_indices = np.random.choice(len(X_train), n_candidates, replace=False)
    candidate_x_train = X_train[candidate_indices]
    candidate_y_train = y_train[candidate_indices]
    
    
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nFeature dimension: {X_train.shape[1]}")
    print(f"Candidate set size: {len(candidate_x_train)}")
    
    
    model = ModernNCA(
        d_in=X_train.shape[1],
        d_out=2,
        dim=args.dim,
        dropout=args.dropout,
        d_block=args.d_block,
        n_blocks=args.n_blocks,
        temperature=args.temperature,
        sample_rate=args.sample_rate
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print("\n" + "=" * 80)
    print(" Training ModernNCA ".center(80, "="))
    print("=" * 80)
    
    best_val_f1 = -1
    patience_counter = 0
    
    for epoch in range(args.epochs):
        
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            candidate_x_tensor = torch.FloatTensor(candidate_x_train).to(device)
            candidate_y_tensor = torch.LongTensor(candidate_y_train).to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x, batch_y, candidate_x_tensor, candidate_y_tensor, is_train=True)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        
        candidate_x_tensor = torch.FloatTensor(candidate_x_train).to(device)
        candidate_y_tensor = torch.LongTensor(candidate_y_train).to(device)
        val_acc, val_prec, val_rec, val_f1, val_auc, val_auprc = evaluate(
            model, val_loader, candidate_x_tensor, candidate_y_tensor, device
        )
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_modernnca_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    
    model.load_state_dict(torch.load('best_modernnca_model.pth'))
    candidate_x_tensor = torch.FloatTensor(candidate_x_train).to(device)
    candidate_y_tensor = torch.LongTensor(candidate_y_train).to(device)
    test_acc, test_prec, test_rec, test_f1, test_auc, test_auprc = evaluate(
        model, test_loader, candidate_x_tensor, candidate_y_tensor, device
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  AUC-ROC:   {test_auc:.4f}")
    print(f"  AUPRC:     {test_auprc:.4f}")
    
    
    if os.path.exists('best_modernnca_model.pth'):
        os.remove('best_modernnca_model.pth')
    
    return {
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'auc': test_auc,
        'auprc': test_auprc
    }
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    candidate_x_train = torch.FloatTensor(X_train).to(device)
    candidate_y_train = torch.LongTensor(y_train).to(device)
    
    model = ModernNCA(
        d_in=X_train.shape[1],
        d_out=2,
        dim=args.dim,
        dropout=args.dropout,
        d_block=args.d_block,
        n_blocks=args.n_blocks,
        temperature=args.temperature,
        sample_rate=args.sample_rate
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"\nTraining ModernNCA...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_f1 = -1
    patience_counter = 0
    
    
    if args.dataset == 'merchant':
        ckpt_name = args.industry.lower().replace("-", "")
    else:
        ckpt_name = args.dataset
    ckpt_path = Config.RESULTS_DIR / f'modernnca_{ckpt_name}_best.pt'
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, candidate_x_train, candidate_y_train, optimizer, device)
        
        val_acc, val_prec, val_rec, val_f1, val_auc, val_auprc = evaluate(
            model, val_loader, candidate_x_train, candidate_y_train, device
        )
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    
    model.load_state_dict(torch.load(ckpt_path))
    
    test_acc, test_prec, test_rec, test_f1, test_auc, test_auprc = evaluate(
        model, test_loader, candidate_x_train, candidate_y_train, device
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  AUC-ROC:   {test_auc:.4f}")
    print(f"  AUPRC:     {test_auprc:.4f}")
    
    
    if args.dataset == 'merchant':
        result_name = args.industry.lower().replace("-", "")
    else:
        result_name = args.dataset
    
    results = pd.DataFrame([{
        'Dataset': info['name'],
        'Accuracy': test_acc,
        'Precision': test_prec,
        'Recall': test_rec,
        'F1-Score': test_f1,
        'AUC-ROC': test_auc,
        'AUPRC': test_auprc,
    }])
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    suffix = ''
    result_filename = f'modernnca_{result_name}{suffix}_results.csv'
    result_path = Config.RESULTS_DIR / result_filename
    results.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()

