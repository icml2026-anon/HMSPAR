import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from configs.config import Config
from utils.seed import set_seed
from utils.metrics import compute_binary_metrics, print_metrics
from data.dataset import MerchantDataset, CDNOWDataset, RetailDataset, InstacartDataset, SalesWeeklyDataset, TaFengDataset
from models.hmspar import HMSPAR
from tqdm.auto import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        ts_data = batch['ts_data'].to(device)
        image = batch['image'].to(device)
        text_embedding = batch['text_embedding'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs, aux_loss = model(ts_data, image, text_embedding, return_aux_loss=True)
        loss = criterion(outputs, labels)
        
        if aux_loss is not None:
            loss = loss + aux_loss
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        del ts_data, image, text_embedding, labels, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds_proba = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            ts_data = batch['ts_data'].to(device)
            image = batch['image'].to(device)
            text_embedding = batch['text_embedding'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(ts_data, image, text_embedding)
            probs = torch.sigmoid(outputs)
            
            all_preds_proba.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_binary_metrics(
        np.array(all_labels),
        np.array(all_preds_proba)
    )
    
    return metrics


def main(args, seed=None):
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(Config.SEED)
    
    print("=" * 80)
    if args.dataset == 'merchant':
        print(f" HMSPAR Training on Merchant Dataset ({args.industry}) ".center(80, "="))
    else:
        raise ValueError(f"This train.py is specialized for merchant dataset only. Use dataset-specific training scripts for other datasets.")
    print("=" * 80)
    
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    params = Config.get_industry_params(args.industry)
    
    if args.learning_rate is not None:
        params['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        params['weight_decay'] = args.weight_decay
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.ts_hidden_dim is not None:
        params['ts_hidden_dim'] = args.ts_hidden_dim
    if args.fusion_dim is not None:
        params['fusion_dim'] = args.fusion_dim
    if args.dropout_rate is not None:
        params['dropout_rate'] = args.dropout_rate
    if args.order is not None:
        params['order'] = args.order
    if args.n_experts is not None:
        params['n_experts'] = args.n_experts
    if args.n_heads is not None:
        params['n_heads'] = args.n_heads
    if args.epochs is not None:
        params['epochs'] = args.epochs
    
    print("\nLoading data...")
    
    if args.dataset == 'merchant':
        df = pd.read_csv(Config.DATA_DIR / 'merchant_data.csv')
        text_embeddings = np.load(Config.DATA_DIR / 'text_embeddings.npy')
        isa_gaf_images = np.load(Config.DATA_DIR / 'isa_gaf_images.npy')
        
        industry_df = df[df['Industry'] == args.industry].copy()
        industry_indices = industry_df.index
        
        print(f"\nIndustry: {args.industry}")
        print(f"Total samples: {len(industry_df)}")
        print(f"Anomaly ratio: {industry_df['is_anomalous'].mean():.2%}")
        
        ts_cols = [col for col in df.columns if col.startswith('txn_')]
        
        train_df, temp_df = train_test_split(
            industry_df, test_size=Config.TEST_RATIO + Config.VAL_RATIO, 
            random_state=Config.SEED, stratify=industry_df['is_anomalous']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=Config.TEST_RATIO / (Config.TEST_RATIO + Config.VAL_RATIO),
            random_state=Config.SEED, stratify=temp_df['is_anomalous']
        )
        
        train_text = text_embeddings[train_df.index]
        val_text = text_embeddings[val_df.index]
        test_text = text_embeddings[test_df.index]
        
        train_gaf = isa_gaf_images[train_df.index]
        val_gaf = isa_gaf_images[val_df.index]
        test_gaf = isa_gaf_images[test_df.index]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        train_dataset = MerchantDataset(
            train_df.reset_index(drop=True),
            ts_cols,
            train_text,
            train_gaf
        )
        val_dataset = MerchantDataset(
            val_df.reset_index(drop=True),
            ts_cols,
            val_text,
            val_gaf
        )
        test_dataset = MerchantDataset(
            test_df.reset_index(drop=True),
            ts_cols,
            test_text,
            test_gaf
        )
        
    # This script is specialized for merchant dataset only
    
    print(f"\nHyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'] * 2,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'] * 2,
        shuffle=False,
        num_workers=0
    )
    
    print("\nInitializing model...")
    
    ts_input_dim = 1
    text_embed_dim = text_embeddings.shape[1]
    image_size = Config.IMAGE_SIZE
    
    model = HMSPAR(
        ts_input_dim=ts_input_dim,
        ts_hidden_dim=params['ts_hidden_dim'],
        text_embed_dim=text_embed_dim,
        fusion_dim=params['fusion_dim'],
        dropout_rate=params['dropout_rate'],
        order=params.get('order', 4),
        n_experts=params.get('n_experts', 4),
        n_heads=params.get('n_heads', 4)
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=params['epochs'],
        eta_min=1e-6
    )
    
    print("\n" + "=" * 80)
    print(" Training Loop ".center(80, "="))
    print("=" * 80)
    
    best_val_f1 = 0
    best_epoch = 0
    best_val_metrics = None
    best_model_state = None
    
    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch + 1}/{params['epochs']}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Val F1:     {val_metrics['f1']:.4f}")
        print(f"  Val AUC:    {val_metrics.get('auc', 0):.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            best_val_metrics = val_metrics.copy()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            checkpoint_path = Config.CHECKPOINT_DIR / f"{args.industry}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"  Saved best model (F1: {best_val_f1:.4f})")
        
        scheduler.step()
    
    print(f"\nBest validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    print("\n" + "=" * 80)
    print(" Test Set Evaluation ".center(80, "="))
    print("=" * 80)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    test_metrics = evaluate(model, test_loader, device)
    print_metrics(test_metrics, prefix="Test ")
    
    results_df = pd.DataFrame([test_metrics])
    results_df['industry'] = args.industry
    results_df['best_epoch'] = best_epoch
    
    results_path = Config.RESULTS_DIR / f"{args.industry}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    if args.output_json:
        output_data = {
            'val_metrics': {k: float(v) if v is not None else None for k, v in best_val_metrics.items()},
            'test_metrics': {k: float(v) if v is not None else None for k, v in test_metrics.items()},
            'best_epoch': best_epoch,
            'params': params
        }
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print(" Training Complete ".center(80, "="))
    print("=" * 80)
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMSPAR on Multiple Datasets")
    parser.add_argument(
        '--dataset',
        type=str,
        default='merchant',
        choices=['merchant'],
        help='Dataset to train on (merchant only in this script)'
    )
    parser.add_argument(
        '--industry',
        type=str,
        required=True,
        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
        help='Industry to train on (required)'
    )
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (for Bayesian optimization)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay (for Bayesian optimization)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (for Bayesian optimization)')
    parser.add_argument('--ts-hidden-dim', type=int, default=None,
                        help='Time series hidden dimension (for Bayesian optimization)')
    parser.add_argument('--fusion-dim', type=int, default=None,
                        help='Fusion dimension (for Bayesian optimization)')
    parser.add_argument('--dropout-rate', type=float, default=None,
                        help='Dropout rate (for Bayesian optimization)')
    parser.add_argument('--order', type=int, default=None,
                        help='Taylor expansion order (for Bayesian optimization)')
    parser.add_argument('--n-experts', type=int, default=None,
                        help='Number of MoK experts (for Bayesian optimization)')
    parser.add_argument('--n-heads', type=int, default=None,
                        help='Number of attention heads (for Bayesian optimization)')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Path to save JSON output (for Bayesian optimization)')
    parser.add_argument('--multi-seed', action='store_true',
                        help='Run with multiple seeds for statistical analysis')
    
    args = parser.parse_args()
    
    if args.dataset != 'merchant':
        parser.error('This script only supports merchant dataset')
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - HMSPAR {args.dataset.upper()}")
        if args.dataset == 'merchant':
            print(f"Industry: {args.industry}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = main(args, seed)
            all_results.append(result)
            
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY (3 SEEDS)")
        print("="*80)
        print(f"Dataset: {args.dataset.upper()}")
        if args.dataset == 'merchant':
            print(f"Industry: {args.industry}")
        print(f"Seeds: {seeds}")
        print()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
        
        print("Test Results (Mean ± Std):")
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