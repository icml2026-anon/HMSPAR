import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config

def process_retail_churn(prediction_horizon=3, churn_threshold=0):
    from configs.config import Config
    from utils.seed import set_seed
    import torch
    from data.modality_converter import RetailModalityConverter
    
    set_seed(Config.SEED)
    
    amount_path = Config.RETAIL_AMOUNT_CSV
    trans_path = Config.RETAIL_TRANS_CSV
    
    if not amount_path.exists() or not trans_path.exists():
        print("Error: Retail datasets not found. Please download first.")
        return
    
    df_amount = pd.read_csv(amount_path)
    df_trans = pd.read_csv(trans_path)
    
    amount_series = df_amount.iloc[:, :-1].values
    trans_series = df_trans.iloc[:, :-1].values
    
    n_samples, n_periods = amount_series.shape
    
    if n_periods <= prediction_horizon:
        print(f"Error: Not enough periods for prediction horizon {prediction_horizon}")
        return
    
    training_periods = n_periods - prediction_horizon
    train_amount = amount_series[:, :training_periods]
    train_trans = trans_series[:, :training_periods]
    
    future_activity = amount_series[:, training_periods:].sum(axis=1)
    churn_labels = (future_activity <= churn_threshold).astype(int)
    
    print(f"Training periods: {training_periods}")
    print(f"Prediction horizon: {prediction_horizon} months")
    print(f"Churn rate: {churn_labels.mean()*100:.2f}%")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    converter = RetailModalityConverter(Config, device)
    converter.image_size = training_periods
    text_embeddings = converter.generate_text_descriptions(train_amount, train_trans)
    isa_gaf_images = converter.generate_isa_gaf_images(train_amount, train_trans)
    
    output_dir = Config.RETAIL_PROCESSED_DIR / 'churn_task'
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'amount_series.npy', train_amount)
    np.save(output_dir / 'trans_series.npy', train_trans)
    np.save(output_dir / 'text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'labels.npy', churn_labels)
    
    print(f"Churn prediction task data saved to {output_dir}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

def process_cdnow_churn(prediction_horizon=3, churn_threshold=0):
    from configs.config import Config
    from utils.seed import set_seed
    import torch
    from data.modality_converter import CDNOWModalityConverter
    
    set_seed(Config.SEED)
    
    amount_path = Config.CDNOW_AMOUNT_CSV
    trans_path = Config.CDNOW_TRANS_CSV
    
    if not amount_path.exists() or not trans_path.exists():
        print("Error: CDNOW datasets not found. Please run process_cdnow.py first.")
        return
    
    df_amount = pd.read_csv(amount_path)
    df_trans = pd.read_csv(trans_path)
    
    amount_series = df_amount.iloc[:, :-1].values
    trans_series = df_trans.iloc[:, :-1].values
    
    n_samples, n_periods = amount_series.shape
    
    if n_periods <= prediction_horizon:
        print(f"Error: Not enough periods for prediction horizon {prediction_horizon}")
        return
    
    training_periods = n_periods - prediction_horizon
    train_amount = amount_series[:, :training_periods]
    train_trans = trans_series[:, :training_periods]
    
    future_activity = amount_series[:, training_periods:].sum(axis=1)
    churn_labels = (future_activity <= churn_threshold).astype(int)
    
    print(f"Training periods: {training_periods}")
    print(f"Prediction horizon: {prediction_horizon} months")
    print(f"Churn rate: {churn_labels.mean()*100:.2f}%")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    converter = CDNOWModalityConverter(Config, device)
    converter.image_size = training_periods
    text_embeddings = converter.generate_text_descriptions(train_amount, train_trans)
    isa_gaf_images = converter.generate_isa_gaf_images(train_amount, train_trans)
    
    output_dir = Config.CDNOW_PROCESSED_DIR / 'churn_task'
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'amount_series.npy', train_amount)
    np.save(output_dir / 'trans_series.npy', train_trans)
    np.save(output_dir / 'text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'labels.npy', churn_labels)
    
    print(f"Churn prediction task data saved to {output_dir}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

def process_instacart_churn(prediction_horizon=4, target_churn_rate=0.35):
    from configs.config import Config
    from utils.seed import set_seed
    import torch
    from data.modality_converter import InstacartModalityConverter
    
    set_seed(Config.SEED)
    
    processed_dir = Config.INSTACART_PROCESSED_DIR
    
    if not processed_dir.exists():
        print("Error: Instacart processed data not found.")
        return
    
    try:
        order_count_series = np.load(processed_dir / 'order_count_series.npy')
        item_count_series = np.load(processed_dir / 'item_count_series.npy')
    except FileNotFoundError:
        print("Error: Instacart series files not found.")
        return
    
    n_samples, n_periods = order_count_series.shape
    
    if n_periods <= prediction_horizon:
        print(f"Error: Not enough periods for prediction horizon {prediction_horizon}")
        return
    
    training_periods = n_periods - prediction_horizon
    train_orders = order_count_series[:, :training_periods]
    train_items = item_count_series[:, :training_periods]
    
    future_activity = order_count_series[:, training_periods:].sum(axis=1)
    churn_threshold = np.percentile(future_activity, target_churn_rate * 100)
    churn_labels = (future_activity <= churn_threshold).astype(int)
    
    print(f"Future activity range: [{future_activity.min()}, {future_activity.max()}]")
    print(f"Churn threshold (percentile {target_churn_rate*100:.0f}%): {churn_threshold}")
    
    print(f"Training periods: {training_periods}")
    print(f"Prediction horizon: {prediction_horizon} weeks")
    print(f"Churn rate: {churn_labels.mean()*100:.2f}%")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    converter = InstacartModalityConverter(Config, device)
    converter.image_size = training_periods
    text_embeddings = converter.generate_text_descriptions(train_orders, train_items)
    isa_gaf_images = converter.generate_isa_gaf_images(train_orders, train_items)
    
    output_dir = processed_dir / 'churn_task'
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'order_count_series.npy', train_orders)
    np.save(output_dir / 'item_count_series.npy', train_items)
    np.save(output_dir / 'text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'labels.npy', churn_labels)
    
    print(f"Churn prediction task data saved to {output_dir}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

def process_sales_weekly_seasonality():
    from configs.config import Config
    from utils.seed import set_seed
    import torch
    from data.modality_converter import SalesWeeklyModalityConverter
    
    set_seed(Config.SEED)
    
    sales_path = Config.SALES_WEEKLY_CSV
    
    if not sales_path.exists():
        print("Error: Sales weekly dataset not found.")
        return
    
    df = pd.read_csv(sales_path)
    sales_series = df.iloc[:, 1:].values
    product_codes = df.iloc[:, 0].values
    
    n_samples, n_weeks = sales_series.shape
    
    seasonality_labels = []
    for i in range(n_samples):
        sales = sales_series[i]
        if sales.sum() == 0:
            seasonality_labels.append(0)
            continue
        
        weekly_avg = []
        for week_idx in range(min(52, n_weeks)):
            week_sales = []
            for w in range(week_idx, n_weeks, 52):
                if w < n_weeks:
                    week_sales.append(sales[w])
            if week_sales:
                weekly_avg.append(np.mean(week_sales))
        
        if len(weekly_avg) == 0:
            seasonality_labels.append(0)
            continue
        
        weekly_std = np.std(weekly_avg)
        weekly_mean = np.mean(weekly_avg)
        coeff_var = weekly_std / (weekly_mean + 1e-8)
        
        seasonality_labels.append(1 if coeff_var > 0.5 else 0)
    
    seasonality_labels = np.array(seasonality_labels)
    
    print(f"Sales series shape: {sales_series.shape}")
    print(f"Seasonal pattern ratio: {seasonality_labels.mean()*100:.2f}%")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    converter = SalesWeeklyModalityConverter(Config, device)
    text_embeddings = converter.generate_text_descriptions(sales_series, product_codes)
    isa_gaf_images = converter.generate_isa_gaf_images(sales_series)
    
    output_dir = Config.SALES_WEEKLY_PROCESSED_DIR / 'seasonality_task'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'sales_series.npy', sales_series)
    np.save(output_dir / 'text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'labels.npy', seasonality_labels)
    
    print(f"Seasonality prediction task data saved to {output_dir}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

def process_tafeng_repurchase():
    from configs.config import Config
    from utils.seed import set_seed
    import torch
    
    set_seed(Config.SEED)
    
    try:
        amount_series = np.load(Config.DATA_DIR / 'tafeng_amount_series.npy')
        trans_series = np.load(Config.DATA_DIR / 'tafeng_trans_series.npy')
        existing_labels = np.load(Config.DATA_DIR / 'tafeng_labels.npy')
    except FileNotFoundError:
        print("Error: Ta-Feng processed data not found.")
        return
    
    n_samples, n_periods = amount_series.shape
    
    if n_periods != 4:
        print(f"Warning: Expected 4 periods for TaFeng, got {n_periods}")
    
    training_periods = 3
    prediction_period = 3
    train_amount = amount_series[:, :training_periods]
    train_trans = trans_series[:, :training_periods]
    
    repurchase_labels = []
    for i in range(n_samples):
        target_month_activity = amount_series[i, prediction_period] if prediction_period < n_periods else 0
        repurchase_labels.append(1 if target_month_activity > 0 else 0)
    
    repurchase_labels = np.array(repurchase_labels)
    
    print(f"Amount series shape: {amount_series.shape}")
    print(f"Transaction series shape: {trans_series.shape}")
    print(f"Training periods: {training_periods} (Nov 2000 - Jan 2001)")
    print(f"Target period: Feb 2001 (period {prediction_period})")
    print(f"Customer return rate: {repurchase_labels.mean()*100:.2f}%")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from data.modality_converter import RetailModalityConverter
    converter = RetailModalityConverter(Config, device)
    converter.image_size = training_periods
    text_embeddings = converter.generate_text_descriptions(train_amount, train_trans)
    isa_gaf_images = converter.generate_isa_gaf_images(train_amount, train_trans)
    
    output_dir = Config.DATA_DIR / 'tafeng_repurchase_task'
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'amount_series.npy', train_amount)
    np.save(output_dir / 'trans_series.npy', train_trans)
    np.save(output_dir / 'text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'labels.npy', repurchase_labels)
    
    print(f"Repurchase prediction task data saved to {output_dir}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

def validate_appendix_tasks():
    tasks = {
        'retail_churn': 'Customer churn prediction (3-month horizon)',
        'cdnow_churn': 'Customer churn prediction (3-month horizon)', 
        'instacart_churn': 'Customer churn prediction (4-week horizon)',
        'sales_seasonality': 'Product seasonality detection',
        'tafeng_repurchase': 'Frequent repurchase pattern detection'
    }
    
    print("=== Appendix Task Validation ===")
    for task_name, description in tasks.items():
        print(f"- {task_name}: {description}")
    
    return tasks

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['retail_churn', 'cdnow_churn', 'instacart_churn', 'sales_seasonality', 'tafeng_repurchase', 'all'], default='all')
    args = parser.parse_args()
    
    if args.task == 'retail_churn' or args.task == 'all':
        process_retail_churn()
    
    if args.task == 'cdnow_churn' or args.task == 'all':
        process_cdnow_churn()
    
    if args.task == 'instacart_churn' or args.task == 'all':
        process_instacart_churn()
    
    if args.task == 'sales_seasonality' or args.task == 'all':
        process_sales_weekly_seasonality()
    
    if args.task == 'tafeng_repurchase' or args.task == 'all':
        process_tafeng_repurchase()
    
    validate_appendix_tasks()
