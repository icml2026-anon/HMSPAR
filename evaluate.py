import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from configs.config import Config
from utils.seed import set_seed
from utils.metrics import compute_binary_metrics, print_metrics
from data.dataset import MerchantDataset
from models.hmspar import HMSPAR


def main(args):
    """Main evaluation function"""
    print("=" * 80)
    print(" HMSPAR Evaluation ".center(80, "="))
    print("=" * 80)
    
    
    set_seed(Config.SEED)
    
    
    device = Config.DEVICE
    print(f"\nDevice: {device}")
    
    
    checkpoint_path = Config.CHECKPOINT_DIR / f"{args.industry}_best.pth"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first!")
        return
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    
    print("\nLoading data...")
    df = pd.read_csv(Config.DATA_DIR / 'merchant_data.csv')
    text_embeddings = np.load(Config.DATA_DIR / 'text_embeddings.npy')
    isa_gaf_images = np.load(Config.DATA_DIR / 'isa_gaf_images.npy')
    
    
    industry_df = df[df['Industry'] == args.industry].copy()
    industry_indices = industry_df.index
    
    print(f"\nIndustry: {args.industry}")
    print(f"Total samples: {len(industry_df)}")
    
    
    params = Config.get_industry_params(args.industry)
    
    
    ts_cols = [col for col in df.columns if col.startswith('txn_')]
    
    
    dataset = MerchantDataset(
        industry_df.reset_index(drop=True),
        ts_cols,
        text_embeddings[industry_indices],
        isa_gaf_images[industry_indices]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=params['batch_size'] * 2,
        shuffle=False,
        num_workers=0
    )
    
    
    print("\nInitializing model...")
    ts_cols = [col for col in df.columns if col.startswith('txn_')]
    model = HMSPAR(
        ts_input_dim=1,
        ts_hidden_dim=params['ts_hidden_dim'],
        text_embed_dim=text_embeddings.shape[1],
        fusion_dim=params['fusion_dim'],
        dropout_rate=params['dropout_rate'],
        seq_len=len(ts_cols)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    print("\nEvaluating...")
    model.eval()
    all_preds_proba = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
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
    
    print_metrics(metrics, prefix=f"{args.industry} ")
    
    print("\n" + "=" * 80)
    print(" Evaluation Complete ".center(80, "="))
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HMSPAR")
    parser.add_argument(
        '--industry',
        type=str,
        required=True,
        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
        help='Industry to evaluate'
    )
    
    args = parser.parse_args()
    main(args)

