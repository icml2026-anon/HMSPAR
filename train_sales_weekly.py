import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import json
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
from data.dataset import SalesWeeklyDataset
from models.hmspar import HMSPAR
from tqdm.auto import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with load balancing loss"""
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
    """Evaluate model"""
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
            
            del ts_data, image, text_embedding, labels, outputs, probs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    all_preds_proba = np.array(all_preds_proba)
    all_labels = np.array(all_labels)
    
    metrics = compute_binary_metrics(all_labels, all_preds_proba)
    return metrics


def load_sales_weekly_data(task='sales_risk_classification'):
    """Load preprocessed sales weekly data"""
    if task == 'seasonality':
        data_dir = Config.SALES_WEEKLY_PROCESSED_DIR / 'seasonality_task'
        task_desc = "seasonality detection"
    else:
        data_dir = Config.SALES_WEEKLY_PROCESSED_DIR
        task_desc = "sales risk classification"
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Sales Weekly processed data not found for {task_desc}. Please run data preprocessing first."
        )
    
    sales_series = np.load(data_dir / 'sales_series.npy')
    text_embeddings = np.load(data_dir / 'text_embeddings.npy')
    isa_gaf_images = np.load(data_dir / 'isa_gaf_images.npy')
    labels = np.load(data_dir / 'labels.npy')
    
    return sales_series, text_embeddings, isa_gaf_images, labels


def create_dataloaders(sales_series, text_embeddings, isa_gaf_images, labels, batch_size):
    """Create train/val/test dataloaders"""
    indices = np.arange(len(labels))
    
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=Config.TEST_RATIO,
        random_state=Config.SEED,
        stratify=labels
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO),
        random_state=Config.SEED,
        stratify=labels[train_val_idx]
    )
    
    train_dataset = SalesWeeklyDataset(
        sales_series[train_idx],
        text_embeddings[train_idx],
        isa_gaf_images[train_idx],
        labels[train_idx]
    )
    
    val_dataset = SalesWeeklyDataset(
        sales_series[val_idx],
        text_embeddings[val_idx],
        isa_gaf_images[val_idx],
        labels[val_idx]
    )
    
    test_dataset = SalesWeeklyDataset(
        sales_series[test_idx],
        text_embeddings[test_idx],
        isa_gaf_images[test_idx],
        labels[test_idx]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def train_sales_weekly_model(args):
    """Main training function for sales weekly dataset"""
    set_seed(Config.SEED)
    
    task = getattr(args, 'task', 'sales_risk_classification')
    task_name = "Seasonality Detection" if task == 'seasonality' else "Sales Risk Classification"
    
    print("=" * 80)
    print(f" Training HMSPAR on Sales Weekly Dataset - {task_name} ".center(80, "="))
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    sales_series, text_embeddings, isa_gaf_images, labels = load_sales_weekly_data(task)
    
    print(f"\nDataset Info:")
    print(f"  Products: {len(labels)}")
    print(f"  Time steps (weeks): {sales_series.shape[1]}")
    print(f"  Label distribution: {np.bincount(labels.astype(int))}")
    print(f"  ISA-GAF image shape: {isa_gaf_images.shape}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        sales_series, text_embeddings, isa_gaf_images, labels, args.batch_size
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    model = HMSPAR(
        ts_input_dim=Config.SALES_WEEKLY_HYPERPARAMS['input_dim'],  # 1 for single time series
        ts_hidden_dim=args.ts_hidden_dim,
        text_embed_dim=text_embeddings.shape[1],
        fusion_dim=args.fusion_dim,
        dropout_rate=args.dropout_rate,
        img_input_channels=2,  # 2 channels: trend GAF + sparsity GAF
        order=args.order,
        n_experts=args.n_experts,
        n_heads=args.n_heads
    ).to(device)
    
    print(f"\nModel parameters:")
    print(f"  Input dim: 1 (sales only)")
    print(f"  Sequence length: {sales_series.shape[1]}")
    print(f"  TS hidden dim: {args.ts_hidden_dim}")
    print(f"  Fusion dim: {args.fusion_dim}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Taylor order: {args.order}")
    print(f"  N experts: {args.n_experts}")
    print(f"  N heads: {args.n_heads}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    best_val_f1 = 0
    best_epoch = 0
    best_val_metrics = None
    best_model_state = None
    results = {
        'train_losses': [],
        'val_metrics': [],
        'test_metrics': None
    }
    
    print("\n" + "=" * 80)
    print(" Training ".center(80, "="))
    print("=" * 80)
    
    checkpoint_dir = Path(Config.CHECKPOINT_DIR) / 'sales_weekly'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Val F1:     {val_metrics['f1']:.4f}")
        print(f"  Val AUC:    {val_metrics.get('auc', 0):.4f}")
        
        results['train_losses'].append(train_loss)
        results['val_metrics'].append(val_metrics)
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            best_val_metrics = val_metrics.copy()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics
            }, checkpoint_dir / 'best_model.pth')
            print(f"  Saved best model (F1: {best_val_f1:.4f})")
        
        scheduler.step()
    
    print("\n" + "=" * 80)
    print(" Testing ".center(80, "="))
    print("=" * 80)
    
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model state from epoch {best_epoch}")
    else:
        print("Using current model state for evaluation")
        
    test_metrics = evaluate(model, test_loader, device)
    results['test_metrics'] = test_metrics
    
    print(f"\nBest model from epoch {best_epoch}")
    print("Test Metrics:")
    print_metrics(test_metrics, prefix="  ")
    
    results_dir = Path(Config.RESULTS_DIR) / 'sales_weekly'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    
    if hasattr(args, 'output_json') and args.output_json:
        output_data = {
            'val_metrics': {k: float(v) if v is not None else None for k, v in best_val_metrics.items()},
            'test_metrics': {k: float(v) if v is not None else None for k, v in test_metrics.items()},
            'best_epoch': best_epoch,
            'hyperparameters': {
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'ts_hidden_dim': args.ts_hidden_dim,
                'fusion_dim': args.fusion_dim,
                'dropout_rate': args.dropout_rate,
                'order': args.order,
                'n_experts': args.n_experts,
                'n_heads': args.n_heads
            }
        }
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train HMSPAR on Sales Weekly dataset')
    
    sales_params = Config.SALES_WEEKLY_HYPERPARAMS
    
    
    parser.add_argument('--batch-size', type=int, default=sales_params['batch_size'], help='Batch size')
    parser.add_argument('--epochs', type=int, default=sales_params['epochs'], help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=sales_params['learning_rate'], help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=sales_params['weight_decay'], help='Weight decay')
    parser.add_argument('--ts-hidden-dim', type=int, default=sales_params['ts_hidden_dim'], help='Time series hidden dimension')
    parser.add_argument('--fusion-dim', type=int, default=sales_params['fusion_dim'], help='Fusion dimension')
    parser.add_argument('--dropout-rate', type=float, default=sales_params['dropout_rate'], help='Dropout rate')
    parser.add_argument('--order', type=int, default=4, help='Taylor expansion order')
    parser.add_argument('--n-experts', type=int, default=4, help='Number of MoK experts')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save JSON output (for Bayesian optimization)')
    parser.add_argument('--task', type=str, default='sales_risk_classification', choices=['sales_risk_classification', 'seasonality'], help='Task type: sales_risk_classification or seasonality detection')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - SALES_WEEKLY {args.task.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            
            original_seed = Config.SEED
            Config.SEED = seed
            set_seed(seed)
            
            
            result = train_sales_weekly_model(args)
            all_results.append(result)
            
            
            Config.SEED = original_seed
            
        
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY (3 SEEDS)")
        print("="*80)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
        
        print(f"Task: {args.task.upper()}")
        print(f"Seeds: {seeds}")
        print("\nTest Results (Mean ± Std):")
        print("-" * 40)
        
        for metric in metrics:
            values = [result['test_metrics'][metric] for result in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample standard deviation
            
            if metric == 'f1':
                print(f"**{metric.upper():>10}: {mean_val:.4f} ± {std_val:.4f}**")
            else:
                print(f"{metric.upper():>12}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\nIndividual Results:")
        print("-" * 40)
        for i, (seed, result) in enumerate(zip(seeds, all_results)):
            print(f"Seed {seed}: F1 = {result['test_metrics']['f1']:.4f}")
            
        
        import json
        from pathlib import Path
        results_dir = Path('results') / f'sales_weekly_{args.task}_multiseed'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        aggregated_results = {
            'seeds': seeds,
            'individual_results': all_results,
            'statistics': {
                metric: {
                    'mean': float(np.mean([r['test_metrics'][metric] for r in all_results])),
                    'std': float(np.std([r['test_metrics'][metric] for r in all_results], ddof=1)),
                    'values': [float(r['test_metrics'][metric]) for r in all_results]
                } for metric in metrics
            }
        }
        
        with open(results_dir / 'multiseed_results.json', 'w') as f:
            json.dump(aggregated_results, f, indent=2)
            
        print(f"\nMulti-seed results saved to: {results_dir}")
        
    else:
        
        train_sales_weekly_model(args)


if __name__ == "__main__":
    main()
