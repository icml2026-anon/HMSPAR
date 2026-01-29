import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from configs.config import Config
from utils.seed import set_seed
from utils.metrics import compute_binary_metrics, print_metrics
from data.dataset import CDNOWDataset
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
            
            del ts_data, image, text_embedding, labels, outputs, probs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    all_preds_proba = np.array(all_preds_proba)
    all_labels = np.array(all_labels)
    
    metrics = compute_binary_metrics(all_labels, all_preds_proba)
    return metrics


def train_cdnow_model(args):
    set_seed(Config.SEED)
    
    task = getattr(args, 'task', 'engagement')
    task_name = "Customer Churn Prediction" if task == 'churn' else "Customer Engagement Level"
    
    print("=" * 80)
    print(f" Training HMSPAR on CDNOW Dataset - {task_name} ".center(80, "="))
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset parameters
    params = Config.CDNOW_HYPERPARAMS.copy()
    
    # Load CDNOW data with task-specific directory
    print("\nLoading CDNOW data...")
    if task == 'churn':
        data_dir = Config.CDNOW_PROCESSED_DIR / 'churn_task'
        task_desc = "churn prediction"
    else:
        data_dir = Config.CDNOW_PROCESSED_DIR
        task_desc = "customer engagement"
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"CDNOW processed data not found for {task_desc}. Please run data preprocessing first."
        )
    
    amount_series = np.load(data_dir / 'amount_series.npy')
    trans_series = np.load(data_dir / 'trans_series.npy')
    text_embeddings = np.load(data_dir / 'text_embeddings.npy')
    isa_gaf_images = np.load(data_dir / 'isa_gaf_images.npy')
    labels = np.load(data_dir / 'labels.npy')
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Positive ratio: {labels.mean():.2%}")
    print(f"  Amount series shape: {amount_series.shape}")
    print(f"  Trans series shape: {trans_series.shape}")
    print(f"  Text embeddings shape: {text_embeddings.shape}")
    print(f"  ISA-GAF images shape: {isa_gaf_images.shape}")
    print(f"  Expected ts_input_dim: 2 (amount + trans stacked by dataset)")
    
    n_samples = len(labels)
    
    # Split indices consistently
    indices = list(range(n_samples))
    
    train_indices, temp_indices = train_test_split(
        indices, test_size=Config.TEST_RATIO + Config.VAL_RATIO,
        random_state=Config.SEED, stratify=labels
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=Config.TEST_RATIO / (Config.TEST_RATIO + Config.VAL_RATIO),
        random_state=Config.SEED, stratify=labels[temp_indices]
    )
    
    # Create datasets using loaded data
    train_dataset = CDNOWDataset(
        amount_series[train_indices], trans_series[train_indices],
        text_embeddings[train_indices], isa_gaf_images[train_indices], labels[train_indices]
    )
    val_dataset = CDNOWDataset(
        amount_series[val_indices], trans_series[val_indices],
        text_embeddings[val_indices], isa_gaf_images[val_indices], labels[val_indices]
    )
    test_dataset = CDNOWDataset(
        amount_series[test_indices], trans_series[test_indices],
        text_embeddings[test_indices], isa_gaf_images[test_indices], labels[test_indices]
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    if args.epochs is not None:
        params['epochs'] = args.epochs
    
    # Override with Bayesian optimization parameters if provided (like train.py)
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
    
    print(f"\nHyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Create data loaders
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
    
    # Get model input dimensions from dataset sample (like train.py)
    sample = train_dataset[0] if hasattr(train_dataset, '__getitem__') else train_dataset.dataset[train_dataset.indices[0]]
    ts_input_dim = sample['ts_data'].shape[0]  # First dimension is input features (2 for amount+trans)
    text_embed_dim = sample['text_embedding'].shape[-1]
    
    model = HMSPAR(
        ts_input_dim=ts_input_dim,
        ts_hidden_dim=params['ts_hidden_dim'],
        text_embed_dim=text_embed_dim,
        fusion_dim=params['fusion_dim'],
        dropout_rate=params['dropout_rate'],
        img_input_channels=isa_gaf_images.shape[1],  # Use actual image channels
        order=params.get('order', 4),
        n_experts=params.get('n_experts', 4),
        n_heads=params.get('n_heads', 4)
    ).to(device)
    
    print(f"\nModel parameters:")
    print(f"  TS input dim: {ts_input_dim}")
    print(f"  Sequence length: {sample['ts_data'].shape[0]}")
    print(f"  TS hidden dim: {params['ts_hidden_dim']}")
    print(f"  Text embed dim: {text_embed_dim}")
    print(f"  Fusion dim: {params['fusion_dim']}")
    print(f"  Dropout rate: {params['dropout_rate']}")
    print(f"  Taylor order: {params.get('order', 4)}")
    print(f"  N experts: {params.get('n_experts', 4)}")
    print(f"  N heads: {params.get('n_heads', 4)}")
    
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
            best_val_metrics = val_metrics
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Save checkpoint to file
            task_suffix = '_churn' if task == 'churn' else ''
            checkpoint_dir = Path(Config.CHECKPOINT_DIR) / f'cdnow{task_suffix}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics,
                'training_args': args,
                'best_metrics': test_metrics if 'test_metrics' in locals() else None
            }, checkpoint_dir / 'best_model.pth')
            print(f"  Saved best model (F1: {best_val_f1:.4f})")
        
        scheduler.step()
    
    print("\n" + "=" * 80)
    print(" Testing ".center(80, "="))
    print("=" * 80)
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"Best model from epoch {best_epoch}")
    
    test_metrics = evaluate(model, test_loader, device)
    print("Test Metrics:")
    print_metrics(test_metrics)
    
    # Save results
    results_dir = Path('results') / f'cdnow_{task}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'best_epoch': best_epoch,
        'best_val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'hyperparameters': params
    }
    
    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save JSON output for Bayesian optimization (like train.py)
    if hasattr(args, 'output_json') and args.output_json:
        output_data = {
            'val_metrics': {k: float(v) if v is not None else None for k, v in best_val_metrics.items()},
            'test_metrics': {k: float(v) if v is not None else None for k, v in test_metrics.items()},
            'best_epoch': best_epoch,
            'hyperparameters': {
                'learning_rate': params['learning_rate'],
                'weight_decay': params['weight_decay'],
                'batch_size': params['batch_size'],
                'ts_hidden_dim': params['ts_hidden_dim'],
                'fusion_dim': params['fusion_dim'],
                'dropout_rate': params['dropout_rate'],
                'order': params.get('order', 4),
                'n_experts': params.get('n_experts', 4),
                'n_heads': params.get('n_heads', 4)
            }
        }
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train HMSPAR on CDNOW dataset')
    
    cdnow_params = Config.CDNOW_HYPERPARAMS
    
    parser.add_argument('--batch-size', type=int, default=cdnow_params['batch_size'], help='Batch size')
    parser.add_argument('--epochs', type=int, default=cdnow_params['epochs'], help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=cdnow_params['learning_rate'], help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=cdnow_params['weight_decay'], help='Weight decay')
    parser.add_argument('--ts-hidden-dim', type=int, default=cdnow_params['ts_hidden_dim'], help='Time series hidden dimension')
    parser.add_argument('--fusion-dim', type=int, default=cdnow_params['fusion_dim'], help='Fusion dimension')
    parser.add_argument('--dropout-rate', type=float, default=cdnow_params['dropout_rate'], help='Dropout rate')
    parser.add_argument('--order', type=int, default=None, help='Taylor expansion order (for Bayesian optimization)')
    parser.add_argument('--n-experts', type=int, default=None, help='Number of MoK experts (for Bayesian optimization)')
    parser.add_argument('--n-heads', type=int, default=None, help='Number of attention heads (for Bayesian optimization)')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save JSON output (for Bayesian optimization)')
    parser.add_argument('--task', type=str, default='engagement', choices=['engagement', 'churn'], help='Task type: engagement or churn prediction')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    # Run with multiple seeds for statistical reporting
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - CDNOW {args.task.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            # Backup original seed setting
            original_seed = Config.SEED
            Config.SEED = seed
            set_seed(seed)
            
            # Run training
            result = train_cdnow_model(args)
            all_results.append(result)
            
            # Restore original seed
            Config.SEED = original_seed
            
        # Calculate statistics
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
            
        # Save aggregated results
        import json
        from pathlib import Path
        results_dir = Path('results') / f'cdnow_{args.task}_multiseed'
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
        # Single seed run
        train_cdnow_model(args)


if __name__ == "__main__":
    main()
