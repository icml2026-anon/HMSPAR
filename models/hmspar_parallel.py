\
\
\

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class SparsityAwareGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.sparsity_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.value_embed = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x, sparsity_mask):
        B, L, D = x.shape
        mask = sparsity_mask.unsqueeze(-1).expand(-1, -1, D)
        embed = torch.where(mask > 0.5, self.value_embed.expand(B, L, -1), self.sparsity_embed.expand(B, L, -1))
        gate = self.gate_proj(x + embed)
        return x * gate + embed * (1 - gate)

class TaylorKANLayer(nn.Module):
    def __init__(self, input_dim, out_dim, order, addbias=True):
        super(TaylorKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias

        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
        self.sparse_gate = SparsityAwareGate(input_dim)

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x, sparsity_mask=None):
        x = x.permute(0, 2, 1)
        B, L, C_in = x.shape

        if sparsity_mask is not None:
            x = self.sparse_gate(x, sparsity_mask)

        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1, -1)

        y = torch.zeros((B, self.out_dim, L), device=x.device, dtype=x.dtype)
        x_power = torch.ones_like(x_expanded)
        for i in range(self.order):
            y += (x_power * self.coeffs[:, :, i].unsqueeze(1)).sum(dim=-1)
            x_power = x_power * x_expanded

        if self.addbias:
            y += self.bias.T

        return y.permute(0, 2, 1)

class ModalityAwareMoE(nn.Module):
    def __init__(self, in_features, out_features, n_modalities=3, n_experts_per_modality=2, top_k=2):
        super().__init__()
        self.n_modalities = n_modalities
        self.n_experts_per_modality = n_experts_per_modality
        self.n_experts = n_modalities * n_experts_per_modality
        self.top_k = min(top_k, self.n_experts)
        self.in_features = in_features
        self.out_features = out_features

        self.router = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, self.n_experts)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(in_features, out_features)
            ) for _ in range(self.n_experts)
        ])

        self.modality_bias = nn.Parameter(torch.zeros(n_modalities, self.n_experts))
        for i in range(n_modalities):
            start_idx = i * n_experts_per_modality
            end_idx = start_idx + n_experts_per_modality
            self.modality_bias.data[i, start_idx:end_idx] = 1.0

        self.noise_std = 0.1

    def forward(self, x, modality_idx=None):
        B = x.shape[0]

        logits = self.router(x)

        if modality_idx is not None and modality_idx < self.n_modalities:
            logits = logits + self.modality_bias[modality_idx].unsqueeze(0)

        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros(B, self.out_features, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            weight = top_k_weights[:, k].unsqueeze(-1)
            for e_idx in range(self.n_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    output[mask] += weight[mask] * self.experts[e_idx](x[mask])

        return output

class MoKLayer(nn.Module):
    def __init__(self, in_features, out_features, expert_config):
        super(MoKLayer, self).__init__()
        self.n_expert = len(expert_config)

        self.gate = nn.Linear(in_features, self.n_expert)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.GELU(),
                nn.Linear(in_features, out_features)
            ) for _ in expert_config
        ])

    def forward(self, x):
        scores = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return torch.einsum("boi,bi->bo", expert_outputs, scores)

class DualChannelEncoder(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.trend_proj = nn.Linear(seq_len, hidden_dim)
        self.sparse_proj = nn.Linear(seq_len, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, trend_channel, sparse_channel):
        B = trend_channel.shape[0]

        trend_feat = self.trend_proj(trend_channel).unsqueeze(1)
        sparse_feat = self.sparse_proj(sparse_channel).unsqueeze(1)

        combined = torch.cat([trend_feat, sparse_feat], dim=1)
        attn_out, _ = self.cross_attn(combined, combined, combined)

        trend_attn = attn_out[:, 0, :]
        sparse_attn = attn_out[:, 1, :]

        gate_input = torch.cat([trend_attn, sparse_attn], dim=-1)
        gate_weight = self.gate(gate_input)

        fused = gate_weight * trend_attn + (1 - gate_weight) * sparse_attn
        return self.norm(fused)

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, order=4, seq_len=13):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.dual_encoder = DualChannelEncoder(seq_len, hidden_dim)

        self.kan_layer1 = TaylorKANLayer(input_dim, hidden_dim, order=order)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.kan_layer2 = TaylorKANLayer(hidden_dim, output_dim, order=order)
        self.norm2 = nn.LayerNorm(output_dim)

        self.sparsity_encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        B, C, L = x.shape

        if C >= 2:
            trend_channel = x[:, 0, :]
            sparse_channel = (x[:, 0, :] > 1e-6).float()
        else:
            trend_channel = x.squeeze(1)
            sparse_channel = (trend_channel > 1e-6).float()

        sparsity_mask = sparse_channel

        h = self.kan_layer1(x, sparsity_mask)
        h = self.norm1(h)
        h = F.gelu(h)

        h = h.permute(0, 2, 1)
        h = self.kan_layer2(h, None)
        h = self.norm2(h)

        h = h.permute(0, 2, 1)
        kan_feat = self.pool(h).squeeze(-1)

        sparse_feat = self.sparsity_encoder(sparsity_mask)

        gate_input = torch.cat([kan_feat, sparse_feat], dim=-1)
        gate = self.fusion_gate(gate_input)

        output = gate * kan_feat + (1 - gate) * sparse_feat

        return output

class ImageEncoder(nn.Module):

    def __init__(self, output_dim, input_channels=2):
\
\
\
\

        super().__init__()

        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        original_weights = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.resnet.conv1.weight.data = (
            original_weights.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1) / 1.5
        )

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
\
\
\
\
\
\
\
\

        return self.resnet(x)

class StaticFeatureEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
\
\
\
\
\

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, x):
\
\
\
\
\
\
\
\

        return self.encoder(x)

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B = query.shape[0]

        q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, -1)
        return self.out_proj(out)

class HeterogeneousModalityFusion(nn.Module):
    def __init__(self, fusion_dim, n_modalities=3, dropout=0.1):
        super().__init__()
        self.n_modalities = n_modalities
        self.fusion_dim = fusion_dim

        self.modality_norms = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in range(n_modalities)])

        self.cross_attns = nn.ModuleList([
            CrossModalAttention(fusion_dim, num_heads=4, dropout=dropout)
            for _ in range(n_modalities)
        ])

        self.modality_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid()
            ) for _ in range(n_modalities)
        ])

        self.global_context = nn.Sequential(
            nn.Linear(fusion_dim * n_modalities, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, n_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, modality_feats):
        B = modality_feats[0].shape[0]

        normed_feats = [self.modality_norms[i](modality_feats[i]) for i in range(self.n_modalities)]

        enhanced_feats = []
        for i in range(self.n_modalities):
            other_feats = [normed_feats[j] for j in range(self.n_modalities) if j != i]
            other_concat = torch.stack(other_feats, dim=1).mean(dim=1)

            cross_feat = self.cross_attns[i](normed_feats[i], other_concat, other_concat)

            gate = self.modality_gates[i](torch.cat([normed_feats[i], cross_feat], dim=-1))
            enhanced = gate * normed_feats[i] + (1 - gate) * cross_feat
            enhanced_feats.append(enhanced)

        global_input = torch.cat(enhanced_feats, dim=-1)
        weights = self.global_context(global_input)

        stacked = torch.stack(enhanced_feats, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        return fused, weights

class HMSPAR(nn.Module):
    def __init__(
        self,
        ts_input_dim,
        ts_hidden_dim,
        text_embed_dim,
        fusion_dim,
        dropout_rate=0.3,
        img_input_channels=2,
        static_input_dim=None,
        seq_len=13
    ):
        super().__init__()

        self.use_static_features = static_input_dim is not None
        self.fusion_dim = fusion_dim

        self.ts_encoder = TimeSeriesEncoder(
            ts_input_dim,
            ts_hidden_dim,
            fusion_dim,
            seq_len=seq_len
        )
        self.img_encoder = ImageEncoder(fusion_dim, input_channels=img_input_channels)

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        if self.use_static_features:
            self.static_encoder = StaticFeatureEncoder(
                static_input_dim,
                fusion_dim,
                dropout_rate
            )
            n_modalities = 4
        else:
            n_modalities = 3

        self.heterogeneous_fusion = HeterogeneousModalityFusion(
            fusion_dim,
            n_modalities=n_modalities,
            dropout=dropout_rate
        )

        self.modality_moe = ModalityAwareMoE(
            fusion_dim,
            fusion_dim,
            n_modalities=n_modalities,
            n_experts_per_modality=2,
            top_k=2
        )

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(fusion_dim // 4, 1)
        )

    def forward(self, ts_data, image, text_embedding, static_features=None):
        feat_ts = self.ts_encoder(ts_data)
        feat_img = self.img_encoder(image)
        feat_text = self.text_proj(text_embedding)

        if self.use_static_features and static_features is not None:
            feat_static = self.static_encoder(static_features)
            modality_feats = [feat_ts, feat_img, feat_text, feat_static]
        else:
            modality_feats = [feat_ts, feat_img, feat_text]

        fused_feat, modality_weights = self.heterogeneous_fusion(modality_feats)

        enhanced_feat = self.modality_moe(fused_feat)

        logits = self.prediction_head(enhanced_feat).squeeze(-1)

        return logits

