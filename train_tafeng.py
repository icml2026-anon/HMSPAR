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
from data.dataset import TaFengDataset
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
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    """Evaluate model and compute metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            ts_data = batch['ts_data'].to(device)
            image = batch['image'].to(device)
            text_embedding = batch['text_embedding'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(ts_data, image, text_embedding)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_binary_metrics(all_labels, all_preds)
    
    return metrics


def load_tafeng_data(task='customer_risk_classification'):
    """Load preprocessed TaFeng data"""
    if task == 'repurchase':
        data_dir = Config.DATA_DIR / 'tafeng_repurchase_task'
        task_desc = "repurchase prediction"
    else:
        data_dir = Config.DATA_DIR
        task_desc = "customer risk classification"
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"TaFeng processed data not found for {task_desc}. Please run data preprocessing first."
        )
    
    amount_series = np.load(data_dir / 'tafeng_amount_series.npy' if task != 'repurchase' else data_dir / 'amount_series.npy')
    trans_series = np.load(data_dir / 'tafeng_trans_series.npy' if task != 'repurchase' else data_dir / 'trans_series.npy')
    text_embeddings = np.load(data_dir / 'tafeng_text_embeddings.npy' if task != 'repurchase' else data_dir / 'text_embeddings.npy')
    isa_gaf_images = np.load(data_dir / 'tafeng_isa_gaf_images.npy' if task != 'repurchase' else data_dir / 'isa_gaf_images.npy')
    labels = np.load(data_dir / 'tafeng_labels.npy' if task != 'repurchase' else data_dir / 'labels.npy')
    
    return amount_series, trans_series, text_embeddings, isa_gaf_images, labels


def create_dataloaders(amount_series, trans_series, text_embeddings, isa_gaf_images, labels, batch_size=32):
    """Create train/validation/test data loaders with proper splits"""
    
    indices = np.arange(len(labels))
    
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=0.4, random_state=Config.SEED, stratify=labels
    )
    
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels, test_size=0.5, random_state=Config.SEED, stratify=temp_labels
    )
    
    train_dataset = TaFengDataset(
        amount_series[train_indices], trans_series[train_indices], 
        text_embeddings[train_indices], isa_gaf_images[train_indices], labels[train_indices]
    )
    val_dataset = TaFengDataset(
        amount_series[val_indices], trans_series[val_indices],
        text_embeddings[val_indices], isa_gaf_images[val_indices], labels[val_indices]
    )
    test_dataset = TaFengDataset(
        amount_series[test_indices], trans_series[test_indices],
        text_embeddings[test_indices], isa_gaf_images[test_indices], labels[test_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_tafeng_model(args):
    """Main training function for TaFeng dataset"""
    set_seed(Config.SEED)
    
    task = getattr(args, 'task', 'customer_risk_classification')
    task_name = "Customer Repurchase Prediction" if task == 'repurchase' else "Customer Risk Classification"
    
    print("=" * 80)
    print(f" Training HMSPAR on TaFeng Dataset - {task_name} ".center(80, "="))
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    amount_series, trans_series, text_embeddings, isa_gaf_images, labels = load_tafeng_data(task)
    
    print(f"\nDataset Info:")
    print(f"  Samples: {len(labels)}")
    print(f"  Time steps (periods): {amount_series.shape[1]}")
    print(f"  Label distribution: {np.bincount(labels.astype(int))}")
    print(f"  ISA-GAF image shape: {isa_gaf_images.shape}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        amount_series, trans_series, text_embeddings, isa_gaf_images, labels, args.batch_size
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    model = HMSPAR(
        ts_input_dim=amount_series.shape[-1] if len(amount_series.shape) > 2 else 2,
        ts_hidden_dim=args.ts_hidden_dim,
        text_embed_dim=text_embeddings.shape[1],
        fusion_dim=args.fusion_dim,
        dropout_rate=args.dropout_rate,
        img_input_channels=isa_gaf_images.shape[1],
        order=args.order,
        n_experts=args.n_experts,
        n_heads=args.n_heads,
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_f1 = 0
    best_val_metrics = None
    best_model_state = None
    best_epoch = 0
    
    print("\nStarting Training...")
    print("-" * 80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print("Val Metrics:")
        print_metrics(val_metrics, prefix="  ")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_metrics = val_metrics.copy()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            
            task_suffix = '_repurchase' if task == 'repurchase' else ''
            checkpoint_dir = Path(Config.CHECKPOINT_DIR) / f'tafeng{task_suffix}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_val_f1,
            }, checkpoint_dir / 'best_model.pth')
    
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*80)
    print(" FINAL RESULTS ".center(80, "="))
    print("="*80)
    print(f"Best Epoch: {best_epoch+1}")
    print("Best Val Metrics:")
    print_metrics(best_val_metrics, prefix="  ")
    print("Test Metrics:")
    print_metrics(test_metrics, prefix="  ")
    
    task_suffix = '_repurchase' if task == 'repurchase' else ''
    results_dir = Path(Config.RESULTS_DIR) / f'tafeng{task_suffix}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'training_results.json', 'w') as f:
        results = {
            'val_metrics': {k: float(v) if v is not None else None for k, v in best_val_metrics.items()},
            'test_metrics': {k: float(v) if v is not None else None for k, v in test_metrics.items()},
            'best_epoch': best_epoch
        }
        json.dump(results, f, indent=2)
    
    if args.output_json:
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
    parser = argparse.ArgumentParser(description='Train HMSPAR on TaFeng dataset')
    
    tafeng_params = Config.TAFENG_HYPERPARAMS
    
    parser.add_argument('--batch-size', type=int, default=tafeng_params['batch_size'], help='Batch size')
    parser.add_argument('--epochs', type=int, default=tafeng_params['epochs'], help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=tafeng_params['learning_rate'], help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=tafeng_params['weight_decay'], help='Weight decay')
    parser.add_argument('--ts-hidden-dim', type=int, default=tafeng_params['ts_hidden_dim'], help='Time series hidden dimension')
    parser.add_argument('--fusion-dim', type=int, default=tafeng_params['fusion_dim'], help='Fusion dimension')
    parser.add_argument('--dropout-rate', type=float, default=tafeng_params['dropout_rate'], help='Dropout rate')
    parser.add_argument('--order', type=int, default=4, help='Taylor expansion order')
    parser.add_argument('--n-experts', type=int, default=4, help='Number of MoK experts')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save JSON output (for Bayesian optimization)')
    parser.add_argument('--task', type=str, default='customer_risk_classification', choices=['customer_risk_classification', 'repurchase'], help='Task type: customer_risk_classification or repurchase prediction')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - TAFENG {args.task.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            
            original_seed = Config.SEED
            Config.SEED = seed
            set_seed(seed)
            
            
            result = train_tafeng_model(args)
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
        results_dir = Path('results') / f'tafeng_{args.task}_multiseed'
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
        
        train_tafeng_model(args)


if __name__ == "__main__":
    main()
