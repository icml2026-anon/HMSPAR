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
from einops import rearrange, repeat
import torch.distributions as D


class VQShapeDataset(Dataset):
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



def log(t, eps=1e-5):
    return t.clamp(min=eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

def onehot_straight_through(p: torch.Tensor):
    max_idx = p.argmax(-1)
    onehot = nn.functional.one_hot(max_idx, p.shape[-1])
    return onehot + p - p.detach()

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_seq_length, embedding_dim)

    def forward(self, x):
        position_indices = torch.arange(0, x.size(1)).long().unsqueeze(0).to(x.device)
        pos_embeddings = self.positional_embeddings(position_indices)
        return pos_embeddings + x


class EuclCodebook(nn.Module):
    """Original VQShape Euclidean Codebook"""
    def __init__(self, num_code: int = 512, dim_code: int = 256, commit_loss=1., entropy_loss=0., entropy_gamma=1.):
        super().__init__()
        self.num_codebook_vectors = num_code
        self.latent_dim = dim_code
        self.commit_loss = commit_loss
        self.entropy_loss = entropy_loss
        self.entropy_gamma = entropy_gamma

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        
        B, N, D = z.shape
        z_flattened = z.view(-1, D)
        
        
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(B, N, D)
        
        
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        
        
        avg_probs = torch.mean(F.one_hot(min_encoding_indices, num_classes=self.num_codebook_vectors).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        
        loss = codebook_loss + self.commit_loss * commitment_loss + self.entropy_loss * entropy_loss
        
        
        z_q = z + (z_q - z).detach()
        
        return z_q, min_encoding_indices.view(B, N), loss


class MLP(nn.Module):
    """Multi-layer perceptron from original VQShape"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class PatchEncoder(nn.Module):
    """Original VQShape PatchEncoder"""
    def __init__(self, dim_embedding: int = 256, patch_size: int = 8, num_patch: int = 64, num_head: int = 6, num_layer: int = 6, input_dim: int = 1):
        super().__init__()

        self.patch_size = patch_size
        self.num_patch = num_patch
        self.input_dim = input_dim
        
        
        patch_features = input_dim * patch_size
        self.input_project = nn.Linear(patch_features, dim_embedding)
        self.pos_embed = PositionalEmbedding(num_patch, dim_embedding)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layer
        )

    def patch_and_embed(self, x):
        
        if x.dim() == 3:
            
            B, seq_len, input_dim = x.shape
            
            
            if seq_len <= self.patch_size:
                
                if seq_len < self.patch_size:
                    
                    patches = []
                    for i in range(self.num_patch):
                        
                        if i % 3 == 0:
                            
                            padded = F.pad(x, (0, 0, 0, self.patch_size - seq_len))
                        elif i % 3 == 1:
                            
                            padded = F.pad(x, (0, 0, 0, self.patch_size - seq_len), mode='reflect')
                        else:
                            
                            padded = F.pad(x, (0, 0, 0, self.patch_size - seq_len), mode='replicate')
                        
                        patches.append(padded.view(B, -1))  # [B, patch_size * input_dim]
                    x = torch.stack(patches, dim=1)  # [B, num_patch, patch_size * input_dim]
                else:
                    
                    patches = []
                    for i in range(self.num_patch):
                        
                        if i == 0:
                            patch = x  # Original
                        elif i < self.num_patch // 2:
                            
                            shift = min(seq_len // 4, i)
                            patch = torch.roll(x, shifts=shift, dims=1)
                        else:
                            
                            patch = x + torch.randn_like(x) * 0.01
                        
                        patches.append(patch.view(B, -1))
                    x = torch.stack(patches, dim=1)  # [B, num_patch, patch_features]
            else:
                
                stride = max(1, (seq_len - self.patch_size) // max(1, self.num_patch - 1))
                patches = []
                
                for i in range(self.num_patch):
                    start_idx = min(i * stride, seq_len - self.patch_size)
                    end_idx = start_idx + self.patch_size
                    patch = x[:, start_idx:end_idx, :]  # [B, patch_size, input_dim]
                    patches.append(patch.view(B, -1))  # [B, patch_size * input_dim]
                
                x = torch.stack(patches, dim=1)  # [B, num_patch, patch_features]
        else:
            
            B, seq_len = x.shape
            
            if seq_len <= self.patch_size:
                
                patches = []
                for i in range(self.num_patch):
                    if seq_len < self.patch_size:
                        if i % 3 == 0:
                            padded = F.pad(x, (0, self.patch_size - seq_len))
                        elif i % 3 == 1:
                            padded = F.pad(x, (0, self.patch_size - seq_len), mode='reflect')
                        else:
                            padded = F.pad(x, (0, self.patch_size - seq_len), mode='replicate')
                    else:
                        if i == 0:
                            padded = x
                        elif i < self.num_patch // 2:
                            shift = min(seq_len // 4, i)
                            padded = torch.roll(x, shifts=shift, dims=1)
                        else:
                            padded = x + torch.randn_like(x) * 0.01
                    
                    patches.append(padded)
                x = torch.stack(patches, dim=1)  # [B, num_patch, patch_size]
            else:
                
                stride = max(1, (seq_len - self.patch_size) // max(1, self.num_patch - 1))
                patches = []
                
                for i in range(self.num_patch):
                    start_idx = min(i * stride, seq_len - self.patch_size)
                    end_idx = start_idx + self.patch_size
                    patch = x[:, start_idx:end_idx]
                    patches.append(patch)
                
                x = torch.stack(patches, dim=1)  # [B, num_patch, patch_size]
        
        x = self.pos_embed(self.input_project(x))
        return x

    def forward(self, x):
        return self.transformer(self.patch_and_embed(x))


class PatchDecoder(nn.Module):
    """Original VQShape PatchDecoder"""
    def __init__(self, dim_embedding: int = 256, patch_size: int = 8, num_head: int = 6, num_layer: int = 6):
        super().__init__()

        self.patch_size = patch_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layer
        )
        self.out_layer = nn.Linear(dim_embedding, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_embedding)/dim_embedding)

    def forward(self, x):
        x = torch.cat([repeat(self.cls_token, '1 1 E -> B 1 E', B=x.shape[0]), x], dim=1)
        out = self.transformer(x)
        x_hat = rearrange(self.out_layer(out[:, 1:, :]), "B L E -> B (L E)")
        return x_hat, out[:, 0, :]


class Tokenizer(nn.Module):
    """Original VQShape Tokenizer"""
    def __init__(self, dim_embedding: int = 256, num_token: int = 32, num_head: int = 6, num_layer: int = 6):
        super().__init__()

        self.tokens = nn.Parameter(torch.randn(1, num_token, dim_embedding)/dim_embedding)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layer
        )

    def forward(self, x, memory_mask=None):
        return self.transformer(repeat(self.tokens, '1 n d -> b n d', b=x.shape[0]), x, memory_key_padding_mask=memory_mask)


class AttributeDecoder(nn.Module):
    """Original VQShape AttributeDecoder"""
    def __init__(self, dim_code: int = 256, dim_embedding: int = 256) -> None:
        super().__init__()

        self.z_head = MLP(dim_embedding, dim_code, dim_embedding)
        self.tl_mean_head = MLP(dim_embedding, 2, dim_embedding)
        self.mu_head = MLP(dim_embedding, 1, dim_embedding)
        self.sigma_head = MLP(dim_embedding, 1, dim_embedding)

    def forward(self, x):
        return (
            self.z_head(x),
            nn.functional.sigmoid(self.tl_mean_head(x)),
            self.mu_head(x),
            nn.functional.softplus(self.sigma_head(x))
        )


class AttributeEncoder(nn.Module):
    """Original VQShape AttributeEncoder"""
    def __init__(self, dim_code: int = 256, dim_embedding: int = 256) -> None:
        super().__init__()
        self.project = nn.Linear(dim_code + 4, dim_embedding)
    
    def forward(self, z, t, l, mu, sigma):
        return self.project(torch.cat([z, mu, sigma, t, l], dim=-1))


def extract_subsequence(x, t, l, norm_length, smooth=9):
    """Sample subsequences specified by t and l from time series x"""
    B, T = x.shape
    relative_positions = torch.linspace(0, 1, steps=norm_length).to(x.device)
    start_indices = (t * (T-1))
    end_indices = (torch.clamp(t + l, max=1) * (T-1))

    grid = start_indices + (end_indices - start_indices) * relative_positions.unsqueeze(0)
    grid = 2.0 * grid / (T - 1) - 1
    grid = torch.stack([grid, torch.ones_like(grid)], dim=-1)

    x = x.unsqueeze(1).unsqueeze(2)
    interpolated = nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return moving_average(interpolated.squeeze(1).squeeze(1), smooth)


def moving_average(x, window_size):
    B, C, _ = x.shape
    filter = torch.ones(C, 1, window_size, device=x.device) / window_size
    padding = window_size // 2
    x = torch.cat([torch.ones(B, C, padding, device=x.device)*x[:, :, [0]], x, torch.ones(B, C, padding, device=x.device)*x[:, :, [-1]]], dim=-1)
    smoothed_x = nn.functional.conv1d(x, filter, groups=C)
    return smoothed_x


def eucl_sim_loss(x, threshold=0.1):
    d = torch.norm(x.unsqueeze(1) - x.unsqueeze(2), dim=-1)
    loss = nn.functional.relu(threshold - d)
    mask = torch.ones_like(loss) - torch.eye(loss.shape[-1], device=loss.device).unsqueeze(0)
    return (loss * mask).mean()


class VQShapeSimplified(nn.Module):
    """Simplified VQShape implementation focusing on core functionality"""
    def __init__(self, input_dim=1, seq_len=512, dim_embedding=128, patch_size=4, num_patch=4,
                 num_enc_head=4, num_enc_layer=2, num_code=64, dim_code=32, num_classes=2):
        super().__init__()
        
        
        if seq_len < 32:
            dim_embedding = min(64, dim_embedding)
            num_patch = max(2, seq_len // 2)
            patch_size = max(1, seq_len // num_patch)
            num_code = min(32, num_code)
            dim_code = min(16, dim_code)
        elif seq_len < 64:
            dim_embedding = min(96, dim_embedding)
            num_patch = max(4, seq_len // 4)
            patch_size = max(2, seq_len // num_patch)
            
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.dim_embedding = dim_embedding
        self.input_dim = input_dim
        
        
        patch_input_dim = patch_size * input_dim
        self.patch_projection = nn.Linear(patch_input_dim, dim_embedding)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch, dim_embedding) * 0.02)
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=find_valid_num_heads(dim_embedding, num_enc_head),
            dim_feedforward=dim_embedding * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layer)
        
        
        self.use_vq = num_code > 0
        if self.use_vq:
            self.codebook = nn.Embedding(num_code, dim_code)
            self.pre_vq_proj = nn.Linear(dim_embedding, dim_code)
            self.post_vq_proj = nn.Linear(dim_code, dim_embedding)
            nn.init.uniform_(self.codebook.weight, -1.0/num_code, 1.0/num_code)
        
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        classifier_input_dim = dim_embedding * 2  # mean + max pooling
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, dim_embedding),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim_embedding, dim_embedding // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_embedding // 2, num_classes)
        )
        
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def create_patches(self, x):
        """Create patches with proper diversity preservation"""
        
        B, seq_len, input_dim = x.shape
        
        if seq_len <= self.patch_size:
            
            patches = []
            for i in range(self.num_patch):
                if seq_len < self.patch_size:
                    
                    if i % 2 == 0:
                        patch = F.pad(x, (0, 0, 0, self.patch_size - seq_len), mode='constant')
                    else:
                        patch = F.pad(x, (0, 0, 0, self.patch_size - seq_len), mode='replicate')
                else:
                    
                    shift = i % seq_len
                    patch = torch.roll(x, shifts=shift, dims=1)
                patches.append(patch.reshape(B, -1))
        else:
            
            stride = max(1, (seq_len - self.patch_size) // max(1, self.num_patch - 1))
            patches = []
            for i in range(self.num_patch):
                start = min(i * stride, seq_len - self.patch_size)
                end = start + self.patch_size
                patch = x[:, start:end, :]
                patches.append(patch.reshape(B, -1))
        
        return torch.stack(patches, dim=1)  # [B, num_patch, patch_size * input_dim]
    
    def vector_quantize(self, x):
        """Simple vector quantization"""
        if not self.use_vq:
            return x, torch.zeros(x.shape[0], x.shape[1], device=x.device)
        
        
        z_e = self.pre_vq_proj(x)  # [B, num_patch, dim_code]
        
        
        distances = torch.cdist(z_e.view(-1, z_e.size(-1)), self.codebook.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices).view_as(z_e)
        
        
        commit_loss = F.mse_loss(z_q.detach(), z_e)
        
        
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.post_vq_proj(z_q)
        
        return z_q, commit_loss
    
    def forward(self, x, mode='classify'):
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, seq_len, 1] for univariate
            
        B, seq_len, input_dim = x.shape
        
        
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - x_mean) / x_std
        
        
        patches = self.create_patches(x)  # [B, num_patch, patch_size * input_dim]
        
        
        embeddings = self.patch_projection(patches)  # [B, num_patch, dim_embedding]
        embeddings = embeddings + self.pos_embedding[:, :self.num_patch, :]
        
        
        encoded = self.transformer(embeddings)  # [B, num_patch, dim_embedding]
        
        
        if self.use_vq:
            quantized, vq_loss = self.vector_quantize(encoded)
        else:
            quantized = encoded
            vq_loss = torch.tensor(0.0, device=x.device)
        
        if mode == 'classify':
            
            pooled_mean = quantized.mean(dim=1)  # [B, dim_embedding]
            pooled_max = quantized.max(dim=1)[0]  # [B, dim_embedding]
            
            
            features = torch.cat([pooled_mean, pooled_max], dim=-1)  # [B, 2*dim_embedding]
            
            logits = self.classifier(features)
            return logits, vq_loss
        else:
            return quantized, None, vq_loss


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
    }


def train_epoch(model, loader, criterion, optimizer, device, vq_loss_weight=0.01):
    model.train()
    total_loss = 0
    total_vq_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits, vq_loss = model(X_batch, mode='classify')
        
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN/inf in logits: {logits}")
            continue
        
        if torch.isnan(vq_loss) or torch.isinf(vq_loss):
            print(f"Warning: NaN/inf in vq_loss: {vq_loss}")
            continue
        
        
        cls_loss = criterion(logits, y_batch.long())
        
        
        seq_len = X_batch.shape[1] if X_batch.dim() == 3 else X_batch.shape[1]
        if seq_len >= 15:  # Long sequences like Instacart
            vq_loss_weight = 0.05  # Moderate VQ weight
        elif seq_len >= 10:  # Medium sequences
            vq_loss_weight = 0.02
        else:  # Short sequences
            vq_loss_weight = 0.01
        
        loss = cls_loss + vq_loss_weight * vq_loss
        
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/inf in total loss: {loss}")
            continue
        
        loss.backward()
        
        
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if total_grad_norm > 100:  # Large gradient
            print(f"Warning: Large gradient norm: {total_grad_norm}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += cls_loss.item()
        total_vq_loss += vq_loss.item()
    
    return total_loss / len(loader), total_vq_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch, mode='classify')
            
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN/inf in evaluation logits")
                continue
            
            
            probs = F.softmax(logits, dim=-1)[:, 1]  # Probability of positive class
                
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_logits.extend(logits.cpu().numpy())
    
    
    if len(all_logits) > 0:
        logits_sample = np.array(all_logits[:5])  # First 5 samples
        probs_sample = np.array(all_probs[:5])
        labels_sample = np.array(all_labels[:5])
        
        print(f"Debug - First 5 samples:")
        print(f"  Logits: {logits_sample}")
        print(f"  Probs: {probs_sample}")
        print(f"  Labels: {labels_sample}")
        print(f"  Logits mean: {np.mean(logits_sample, axis=0)}")
        print(f"  Logits std: {np.std(logits_sample, axis=0)}")
        
        
        prob_std = np.std(all_probs)
        print(f"  Probability std across all samples: {prob_std}")
        if prob_std < 1e-6:
            print("  WARNING: All probabilities are nearly identical - model may not be learning!")
    
    return np.array(all_labels), np.array(all_probs)


def find_valid_num_heads(dim_embedding, desired_heads):
    """Find the largest number of heads <= desired_heads that divides dim_embedding"""
    for heads in range(min(desired_heads, dim_embedding), 0, -1):
        if dim_embedding % heads == 0:
            return heads
    return 1

def main(args, seed=None):
    print("=" * 80)
    print(" VQShape Baseline ".center(80, "="))
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
    
    
    
    if X_train.ndim == 2:
        input_dim, seq_len = 1, X_train.shape[1]
    else:
        seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    
    
    if seq_len < 32:
        print(f"\nAdjusting parameters for short sequence dataset (seq_len={seq_len})...")
        args.dim_embedding = 64
        args.patch_size = max(1, seq_len // 4)
        args.num_patch = max(2, seq_len // args.patch_size)
        args.num_codes = 32
        args.dim_code = 16
        print(f"  Adjusted dim_embedding: {args.dim_embedding}")
        print(f"  Adjusted patch_size: {args.patch_size}")
        print(f"  Adjusted num_patch: {args.num_patch}")
        print(f"  Adjusted num_codes: {args.num_codes}")
        print(f"  Adjusted dim_code: {args.dim_code}")
    elif seq_len < 64:
        print(f"\nAdjusting parameters for medium sequence dataset (seq_len={seq_len})...")
        args.dim_embedding = 96
        args.patch_size = max(2, seq_len // 6)
        args.num_patch = max(4, seq_len // args.patch_size)
        args.num_codes = 64
        print(f"  Adjusted dim_embedding: {args.dim_embedding}")
        print(f"  Adjusted patch_size: {args.patch_size}")
        print(f"  Adjusted num_patch: {args.num_patch}")
        print(f"  Adjusted num_codes: {args.num_codes}")
    else:
        print(f"\nUsing standard parameters for long sequence dataset (seq_len={seq_len})...")
        args.patch_size = min(8, seq_len // 8)
        args.num_patch = min(16, seq_len // args.patch_size)
        print(f"  patch_size: {args.patch_size}")
        print(f"  num_patch: {args.num_patch}")
    
    
    train_dataset = VQShapeDataset(X_train, y_train)
    val_dataset = VQShapeDataset(X_val, y_val)
    test_dataset = VQShapeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training VQShape ".center(80, "="))
    print("=" * 80)
    
    
    valid_heads = find_valid_num_heads(args.dim_embedding, args.num_heads)
    model = VQShapeSimplified(
        input_dim=input_dim,
        seq_len=seq_len,
        dim_embedding=args.dim_embedding,
        patch_size=args.patch_size,
        num_patch=args.num_patch,
        num_enc_head=valid_heads,
        num_enc_layer=args.num_enc_layers,
        num_code=args.num_codes,
        dim_code=args.dim_code,
        num_classes=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model config: input_dim={input_dim}, seq_len={seq_len}")
    print(f"              patch_size={model.patch_size}, num_patch={model.num_patch}")
    print(f"              dim_embedding={model.dim_embedding}, use_vq={model.use_vq}")
    if model.use_vq:
        print(f"              num_codes={args.num_codes}, dim_code={args.dim_code}")
    print(f"Total parameters: {total_params:,}")
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_vq_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.vq_loss_weight)
        scheduler.step()
        
        
        val_labels, val_probs = evaluate(model, val_loader, device)
        val_preds = (val_probs > 0.5).astype(int)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, VQ Loss={train_vq_loss:.4f}, Val F1={val_metrics['f1']:.4f}")
        
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    
    print("\n" + "=" * 80)
    print(" Evaluation ".center(80, "="))
    print("=" * 80)
    
    test_labels, test_probs = evaluate(model, test_loader, device)
    test_preds = (test_probs > 0.5).astype(int)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    
    if args.dataset == 'merchant':
        result_name = args.industry.lower().replace("-", "")
    else:
        result_name = args.dataset
    
    results_df = pd.DataFrame([test_metrics])
    results_df['dataset'] = info['name']
    results_df['model'] = 'VQShape'
    results_df['dim_embedding'] = model.dim_embedding
    results_df['use_vq'] = model.use_vq
    
    suffix = ''
    results_path = Config.RESULTS_DIR / f"vqshape_{result_name}{suffix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)
    
    return test_metrics

def run_vqshape_single_seed(args, seed):
    """Run VQShape with a specific seed and return results"""
    
    return main(args, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQShape Baseline for Time Series Classification')
    parser.add_argument('--dataset', type=str, default='merchant',
                       choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                       choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                       choices=['churn', 'seasonality', 'repurchase'])
    
    
    parser.add_argument('--dim-embedding', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--patch-size', type=int, default=8, help='Patch size')
    parser.add_argument('--num-patch', type=int, default=32, help='Number of patches')
    parser.add_argument('--num-heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--num-enc-layers', type=int, default=3, help='Encoder layers')
    parser.add_argument('--num-tokenizer-layers', type=int, default=3, help='Tokenizer layers')
    parser.add_argument('--num-tokens', type=int, default=16, help='Number of shape tokens')
    parser.add_argument('--num-codes', type=int, default=256, help='Codebook size')
    parser.add_argument('--dim-code', type=int, default=128, help='Code dimension')
    parser.add_argument('--commitment-loss', type=float, default=1.0, help='VQ commitment loss weight')
    
    
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--vq-loss-weight', type=float, default=0.01, help='VQ loss weight')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')
        
    
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - VQSHAPE {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_vqshape_single_seed(args, seed)
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
