\
\
\

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TaylorKANLayer(nn.Module):
    """Taylor expansion-based KAN layer for time series processing"""

    def __init__(self, input_dim, out_dim, order, addbias=True):
        """
        Args:
            input_dim (int): Input dimension
            out_dim (int): Output dimension
            order (int): Taylor expansion order
            addbias (bool): Whether to add bias term
        """
        super(TaylorKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias

        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, input_dim]

        Returns:
            torch.Tensor: Output tensor [batch, seq_len, out_dim]
        """
        x = x.permute(0, 2, 1)
        B, L, C_in = x.shape

        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1, -1)

        y = torch.zeros((B, self.out_dim, L), device=x.device)
        for i in range(self.order):
            y += (x_expanded ** i * self.coeffs[:, :, i].unsqueeze(1)).sum(dim=-1)

        if self.addbias:
            y += self.bias.T

        return y.permute(0, 2, 1)

class MoKLayer(nn.Module):
    """Basic Mixture of Experts layer """
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

class PLELayer(nn.Module):
    """
    Progressive Layered Extraction (PLE) style MoE for personalized multi-modal fusion.

    Each modality has its own specific experts + shared experts across modalities.
    Gating is based on the modality's own features for personalized routing.
    Features:
        - Top-K sparse activation: only activates top_k experts per sample
        - Load balancing loss to prevent expert collapse
        - Noise for exploration during training
    """

    def __init__(self, feature_dim, n_shared_experts=2, n_specific_experts=2, n_modalities=3,
                 top_k=2, load_balance_weight=0.01, noise_std=0.1):
        """
        Args:
            feature_dim (int): Feature dimension for each modality
            n_shared_experts (int): Number of shared experts across modalities
            n_specific_experts (int): Number of specific experts per modality
            n_modalities (int): Number of modalities (default: 3 for ts, image, text)
            top_k (int): Number of experts to activate per sample (sparse activation)
            load_balance_weight (float): Weight for load balancing loss
            noise_std (float): Standard deviation of noise added to gate logits during training
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.n_shared_experts = n_shared_experts
        self.n_specific_experts = n_specific_experts
        self.n_modalities = n_modalities
        self.total_experts = n_shared_experts + n_specific_experts
        self.top_k = min(top_k, self.total_experts)
        self.load_balance_weight = load_balance_weight
        self.noise_std = noise_std

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(n_shared_experts)
        ])

        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.GELU(),
                    nn.Linear(feature_dim, feature_dim)
                ) for _ in range(n_specific_experts)
            ]) for _ in range(n_modalities)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Linear(feature_dim // 2, self.total_experts)
            ) for _ in range(n_modalities)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(n_modalities)
        ])

    def _compute_load_balance_loss(self, gate_logits_list, top_k_masks_list):
        """
        Compute load balancing loss to encourage uniform expert utilization.

        Uses the auxiliary loss from Switch Transformer adapted for Top-K:
        L_aux = n_experts * sum(f_i * P_i) where:
        - f_i = fraction of tokens where expert i is in top-k
        - P_i = fraction of router probability allocated to expert i

        Args:
            gate_logits_list: List of gate logits [B, n_experts] for each modality
            top_k_masks_list: List of top-k masks [B, n_experts] for each modality

        Returns:
            torch.Tensor: Scalar load balancing loss
        """
        total_loss = 0.0

        for gate_logits, top_k_mask in zip(gate_logits_list, top_k_masks_list):

            f = top_k_mask.float().mean(dim=0)

            P = F.softmax(gate_logits, dim=-1).mean(dim=0)

            aux_loss = self.total_experts * (f * P).sum()
            total_loss += aux_loss

        return total_loss / len(gate_logits_list) * self.load_balance_weight

    def _sparse_gating(self, gate_logits):
        """
        Apply Top-K sparse gating with optional noise during training.

        Args:
            gate_logits: [B, n_experts] raw gate logits

        Returns:
            gate_scores: [B, n_experts] sparse gate scores (only top_k non-zero)
            top_k_mask: [B, n_experts] boolean mask of selected experts
        """

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        top_k_values, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)

        top_k_mask = torch.zeros_like(gate_logits, dtype=torch.bool)
        top_k_mask.scatter_(1, top_k_indices, True)

        masked_logits = gate_logits.masked_fill(~top_k_mask, float('-inf'))
        gate_scores = F.softmax(masked_logits, dim=-1)

        return gate_scores, top_k_mask

    def forward(self, modality_features, return_aux_loss=True):
        """
        Forward pass with Top-K sparse expert routing per modality.

        Args:
            modality_features (list): List of [batch, feature_dim] tensors for each modality
            return_aux_loss (bool): Whether to return auxiliary load balancing loss

        Returns:
            tuple: (enhanced_features, aux_loss) if return_aux_loss else enhanced_features
                - enhanced_features: List of [batch, feature_dim] tensors
                - aux_loss: Scalar load balancing loss
        """
        enhanced_features = []
        gate_logits_list = []
        top_k_masks_list = []
        gate_scores_list = []

        for m_idx, feat in enumerate(modality_features):

            gate_logits = self.gates[m_idx](feat)
            gate_logits_list.append(gate_logits)

            gate_scores, top_k_mask = self._sparse_gating(gate_logits)
            top_k_masks_list.append(top_k_mask)
            gate_scores_list.append(gate_scores)

            expert_outputs = []

            for expert in self.shared_experts:
                expert_outputs.append(expert(feat))

            for expert in self.specific_experts[m_idx]:
                expert_outputs.append(expert(feat))

            expert_outputs = torch.stack(expert_outputs, dim=-1)
            weighted_output = torch.einsum("bdi,bi->bd", expert_outputs, gate_scores)

            enhanced = self.layer_norms[m_idx](feat + weighted_output)
            enhanced_features.append(enhanced)

        self.gate_logits_list = gate_logits_list
        self.top_k_masks_list = top_k_masks_list
        self.gate_scores_list = gate_scores_list

        if return_aux_loss:
            aux_loss = self._compute_load_balance_loss(gate_logits_list, top_k_masks_list)
            return enhanced_features, aux_loss

        return enhanced_features

class TimeSeriesEncoder(nn.Module):
    """
    Time Series Encoder using Taylor-KAN for non-linear feature transformation.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, order=4, n_heads=4, max_seq_len=64):
        """
        Args:
            input_dim (int): Input dimension (number of features per time step)
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
            order (int): Taylor expansion order
            n_heads (int): Number of attention heads (unused, kept for compatibility)
            max_seq_len (int): Maximum sequence length (unused, kept for compatibility)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.kan_layer1 = TaylorKANLayer(hidden_dim, hidden_dim, order=order)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.kan_layer2 = TaylorKANLayer(hidden_dim, output_dim, order=order)
        self.norm2 = nn.LayerNorm(output_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Encode time series using Taylor-KAN

        Args:
            x (torch.Tensor): Input time series [batch, input_dim, seq_len]

        Returns:
            torch.Tensor: Encoded features [batch, output_dim]
        """
        B, C, L = x.shape

        x = x.permute(0, 2, 1)

        x = self.input_proj(x)

        x = self.kan_layer1(x.permute(0, 2, 1))
        x = self.norm1(x)
        x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = self.kan_layer2(x)
        x = self.norm2(x)

        x = x.permute(0, 2, 1)
        output = self.pool(x).squeeze(-1)

        return output

class ImageEncoder(nn.Module):
    """Image encoder using ResNet18"""

    def __init__(self, output_dim, input_channels=2):
        """
        Args:
            output_dim (int): Output feature dimension
            input_channels (int): Number of input channels (default: 2 for ISA-GAF)
        """
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
        """
        Encode image

        Args:
            x (torch.Tensor): Input image [batch, 2, H, W]

        Returns:
            torch.Tensor: Encoded features [batch, output_dim]
        """
        return self.resnet(x)

class StaticFeatureEncoder(nn.Module):
    """Static feature encoder for categorical and normalized features"""

    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        """
        Args:
            input_dim (int): Input dimension (number of static features)
            output_dim (int): Output dimension
            dropout_rate (float): Dropout rate
        """
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
        """
        Encode static features

        Args:
            x (torch.Tensor): Static features [batch, input_dim]

        Returns:
            torch.Tensor: Encoded features [batch, output_dim]
        """
        return self.encoder(x)

class HMSPAR(nn.Module):
    """
    HMSPAR: Homologous Multi-modal Sparse Pattern Anomaly Recognition

    Architecture:
    - Time series encoder (Taylor-KAN)
    - Static feature encoder (for UCI dataset)
    - Image encoder (ResNet18)
    - Text projection
    - PLE-style personalized MoE fusion (replaces two-stage MoE)
    - Binary classification head
    """

    def __init__(
        self,
        ts_input_dim,
        ts_hidden_dim,
        text_embed_dim,
        fusion_dim,
        dropout_rate=0.3,
        img_input_channels=2,
        static_input_dim=None,
        order=4,
        n_experts=4,
        n_heads=4,
        n_shared_experts=2,
        n_specific_experts=2
    ):
        """
        Args:
            ts_input_dim (int): Time series input dimension
            ts_hidden_dim (int): Time series hidden dimension
            text_embed_dim (int): Text embedding dimension
            fusion_dim (int): Fusion layer dimension
            dropout_rate (float): Dropout rate
            img_input_channels (int): Number of image input channels (default: 2)
            static_input_dim (int): Static features input dimension (None for merchant data)
            order (int): Taylor expansion order for KAN layers (default: 4)
            n_experts (int): Number of experts in MoK layer (default: 4)
            n_heads (int): Number of attention heads in time series encoder (default: 4)
            n_shared_experts (int): Number of shared experts in PLE (default: 2)
            n_specific_experts (int): Number of specific experts per modality in PLE (default: 2)
        """
        super().__init__()

        self.use_static_features = static_input_dim is not None
        self.n_experts = n_experts
        self.fusion_dim = fusion_dim

        self.ts_encoder = TimeSeriesEncoder(
            ts_input_dim,
            ts_hidden_dim,
            fusion_dim,
            order=order,
            n_heads=n_heads
        )
        self.img_encoder = ImageEncoder(fusion_dim, input_channels=img_input_channels)
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
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

        self.ple_fusion = PLELayer(
            feature_dim=fusion_dim,
            n_shared_experts=n_shared_experts,
            n_specific_experts=n_specific_experts,
            n_modalities=n_modalities
        )

        final_fusion_input_dim = fusion_dim * n_modalities
        self.final_fusion = nn.Sequential(
            nn.Linear(final_fusion_input_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, 1)
        )

    def forward(self, ts_data, image, text_embedding, static_features=None, return_aux_loss=False):
        """
        Forward pass with PLE-style personalized fusion

        Args:
            ts_data (torch.Tensor): Time series data [batch, 1, seq_len]
            image (torch.Tensor): Image data [batch, C, H, W] where C=2 or 4
            text_embedding (torch.Tensor): Text embedding [batch, text_dim]
            static_features (torch.Tensor, optional): Static features [batch, static_dim]
            return_aux_loss (bool): Whether to return auxiliary load balancing loss

        Returns:
            torch.Tensor or tuple: Logits [batch], or (logits, aux_loss) if return_aux_loss
        """

        feat_ts = self.ts_encoder(ts_data)
        feat_img = self.img_encoder(image)
        feat_text = self.text_proj(text_embedding)

        if self.use_static_features and static_features is not None:

            feat_static = self.static_encoder(static_features)
            modality_features = [feat_static, feat_ts, feat_img, feat_text]
        else:

            modality_features = [feat_ts, feat_img, feat_text]

        enhanced_features, aux_loss = self.ple_fusion(modality_features, return_aux_loss=True)

        fused_feat = torch.cat(enhanced_features, dim=1)

        fused_feat = self.final_fusion(fused_feat)

        logits = self.prediction_head(fused_feat).squeeze(-1)

        if return_aux_loss:
            return logits, aux_loss
        return logits

