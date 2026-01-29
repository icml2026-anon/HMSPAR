import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json
from tqdm.auto import tqdm
from configs.config import Config
from models.hmspar import HMSPAR
from data.dataset import InstacartDataset
from utils.metrics import compute_binary_metrics, print_metrics
from utils.seed import set_seed
from utils.focal_loss import FocalLoss


def create_dataloaders(order_count, item_count, text_embeddings, isa_gaf_images, labels, batch_size):

    

    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=Config.TEST_RATIO, random_state=Config.SEED, stratify=labels
    )
    

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO),
        random_state=Config.SEED,
        stratify=labels[train_val_idx]
    )
    

    train_dataset = InstacartDataset(
        order_count[train_idx],
        item_count[train_idx],
        text_embeddings[train_idx],
        isa_gaf_images[train_idx],
        labels[train_idx]
    )
    
    val_dataset = InstacartDataset(
        order_count[val_idx],
        item_count[val_idx],
        text_embeddings[val_idx],
        isa_gaf_images[val_idx],
        labels[val_idx]
    )
    
    test_dataset = InstacartDataset(
        order_count[test_idx],
        item_count[test_idx],
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


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch with load balancing loss"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        ts_data = batch['ts_data'].to(device)
        image = batch['image'].to(device)
        text_embedding = batch['text_embedding'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        output, aux_loss = model(ts_data, image, text_embedding, return_aux_loss=True)
        loss = criterion(output, label)
        
     
        if aux_loss is not None:
            loss = loss + aux_loss
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    all_preds_proba = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            ts_data = batch['ts_data'].to(device)
            image = batch['image'].to(device)
            text_embedding = batch['text_embedding'].to(device)
            label = batch['label'].to(device)
            
            output = model(ts_data, image, text_embedding)
            probs = torch.sigmoid(output)
            
            all_preds_proba.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    metrics = compute_binary_metrics(
        np.array(all_labels),
        np.array(all_preds_proba)
    )
    
    return metrics


def train_instacart_model(args):
 
    
    set_seed(Config.SEED)
    device = Config.DEVICE
    
    task = getattr(args, 'task', 'engagement')
    task_name = "Customer Churn Prediction" if task == 'churn' else "Customer Engagement Level"
    
    print("=" * 80)
    print(f" Training HMSPAR on Instacart Dataset - {task_name} ".center(80, "="))
    print("=" * 80)
    print(f"Device: {device}")
    
    
    print("\nLoading processed data...")
    if task == 'churn':
        data_dir = Config.INSTACART_PROCESSED_DIR / 'churn_task'
    else:
        data_dir = Config.INSTACART_PROCESSED_DIR
    
    order_count_series = np.load(data_dir / 'order_count_series.npy')
    item_count_series = np.load(data_dir / 'item_count_series.npy')
    text_embeddings = np.load(data_dir / 'text_embeddings.npy')
    isa_gaf_images = np.load(data_dir / 'isa_gaf_images.npy')
    labels = np.load(data_dir / 'labels.npy')
    
    print(f"\nDataset Info:")
    print(f"  Samples: {len(labels)}")
    print(f"  Time steps: {order_count_series.shape[1]}")
    print(f"  Label distribution: {np.bincount(labels.astype(int))}")
    
    
    params = {
        'learning_rate': getattr(args, 'learning_rate', Config.INSTACART_HYPERPARAMS['learning_rate']),
        'weight_decay': getattr(args, 'weight_decay', Config.INSTACART_HYPERPARAMS['weight_decay']),
        'epochs': getattr(args, 'epochs', Config.INSTACART_HYPERPARAMS['epochs']),
        'batch_size': getattr(args, 'batch_size', Config.INSTACART_HYPERPARAMS['batch_size']),
        'ts_hidden_dim': getattr(args, 'ts_hidden_dim', Config.INSTACART_HYPERPARAMS['ts_hidden_dim']),
        'fusion_dim': getattr(args, 'fusion_dim', Config.INSTACART_HYPERPARAMS['fusion_dim']),
        'dropout_rate': getattr(args, 'dropout_rate', Config.INSTACART_HYPERPARAMS['dropout_rate']),
        'input_dim': Config.INSTACART_HYPERPARAMS['input_dim'],
        'seq_len': Config.INSTACART_HYPERPARAMS['seq_len'],
        'order': getattr(args, 'order', 4),
        'n_experts': getattr(args, 'n_experts', 4),
        'n_heads': getattr(args, 'n_heads', 4)
    }
    
    print(f"\nHyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    
    train_loader, val_loader, test_loader = create_dataloaders(
        order_count_series, item_count_series, text_embeddings, isa_gaf_images, labels, params['batch_size']
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    model = HMSPAR(
        ts_input_dim=params['input_dim'],
        ts_hidden_dim=params['ts_hidden_dim'],
        text_embed_dim=text_embeddings.shape[1],
        fusion_dim=params['fusion_dim'],
        dropout_rate=params['dropout_rate'],
        img_input_channels=isa_gaf_images.shape[1],
        order=params['order'],
        n_experts=params['n_experts'],
        n_heads=params['n_heads']
    ).to(device)
    
    
    
    class_counts = np.bincount(labels.astype(int))
    
    alpha = class_counts[0] / len(labels)  # ~0.65 for instacart engagement
    print(f"  Focal Loss alpha: {alpha:.4f} (dynamic based on class distribution)")
    print(f"  Focal Loss gamma: 2.0")
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'], eta_min=1e-6)
    
    
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    task_suffix = '_churn' if task == 'churn' else ''
    checkpoint_dir = Path(Config.CHECKPOINT_DIR) / f'instacart{task_suffix}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0
    best_epoch = 0
    best_val_metrics = None
    best_model_state = None
    results = {'train_losses': [], 'val_metrics': []}
    
    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch + 1}/{params['epochs']}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        results['train_losses'].append(train_loss)
        results['val_metrics'].append(val_metrics)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val F1:     {val_metrics['f1']:.4f}")
        print(f"  Val AUC:    {val_metrics['auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            best_val_metrics = val_metrics.copy()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
                'params': params,  # Save hyperparameters for model reconstruction
            }, checkpoint_dir / 'best_model.pth')
            
            print(f"  Saved best model (F1: {best_f1:.4f})")
        
        scheduler.step()
    
    
    print("\n" + "=" * 80)
    print("Testing")
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
    
    task_suffix = '_churn' if task == 'churn' else ''
    results_dir = Path(Config.RESULTS_DIR) / f'instacart{task_suffix}'
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
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train HMSPAR on Instacart dataset')
    
    parser.add_argument('--learning-rate', type=float, default=Config.INSTACART_HYPERPARAMS['learning_rate'])
    parser.add_argument('--weight-decay', type=float, default=Config.INSTACART_HYPERPARAMS['weight_decay'])
    parser.add_argument('--epochs', type=int, default=Config.INSTACART_HYPERPARAMS['epochs'])
    parser.add_argument('--batch-size', type=int, default=Config.INSTACART_HYPERPARAMS['batch_size'])
    parser.add_argument('--ts-hidden-dim', type=int, default=Config.INSTACART_HYPERPARAMS['ts_hidden_dim'])
    parser.add_argument('--fusion-dim', type=int, default=Config.INSTACART_HYPERPARAMS['fusion_dim'])
    parser.add_argument('--dropout-rate', type=float, default=Config.INSTACART_HYPERPARAMS['dropout_rate'])
    
    parser.add_argument('--order', type=int, default=4, help='Taylor expansion order')
    parser.add_argument('--n-experts', type=int, default=4, help='Number of MoK experts')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save JSON output (for Bayesian optimization)')
    parser.add_argument('--task', type=str, default='engagement', choices=['engagement', 'churn'], help='Task type: engagement or churn prediction')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - INSTACART {args.task.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            
            original_seed = Config.SEED
            Config.SEED = seed
            set_seed(seed)
            
            
            result = train_instacart_model(args)
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
        results_dir = Path('results') / f'instacart_{args.task}_multiseed'
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
        
        train_instacart_model(args)


if __name__ == "__main__":
    main()
