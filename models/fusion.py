# ==============================================================================
# models/fusion.py - Multi-Omics Fusion Modules (Concat, Weighted, Attention, Ablation)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionFusion(nn.Module):
    """
    Self-attention gating mechanism between RNA and Auxiliary omics embeddings.
    Computes spot-specific attention weights dynamically.
    """
    def __init__(self, channels: int):
        super(AttentionFusion, self).__init__()
        self.score_net = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Tanh(),
            nn.Linear(channels, 2),
            nn.Softmax(dim=-1)
        )
        self.proj = nn.Linear(channels, channels)

    def forward(self, rna_emb: torch.Tensor, aux_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: (N, 2 * C)
        combined = torch.cat([rna_emb, aux_emb], dim=1)
        # Shape: (N, 2)
        weights = self.score_net(combined)
        w_rna = weights[:, 0].unsqueeze(1)
        w_aux = weights[:, 1].unsqueeze(1)

        fused = w_rna * rna_emb + w_aux * aux_emb
        fused = self.proj(fused)
        return fused, weights


class MultiOmicsFusionModule(nn.Module):
    """
    Modular Multi-Omics Fusion supporting:
      1. 'concat': Linear concatenation MLP (Baseline Muse-GCN)
      2. 'weighted': Explicit modality weighting: Z_joint = alpha * Z_RNA + (1 - alpha) * Z_Aux
      3. 'attention': Dynamic per-spot attention fusion
      4. 'rna_only': Modality ablation (RNA only)
      5. 'aux_only': Modality ablation (Auxiliary omics only)
    """
    def __init__(
        self,
        out_channels: int,
        fusion_type: str = 'concat',
        alpha: float = 0.5,
        learnable_alpha: bool = False
    ):
        super(MultiOmicsFusionModule, self).__init__()
        self.out_channels = out_channels
        self.fusion_type = fusion_type.lower()
        self.alpha_val = alpha

        # Within-modality fusion (Similarity + Spatial)
        self.fusion_layer1 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))
        self.fusion_layer_aux = nn.Sequential(nn.Linear(2 * out_channels, out_channels))

        # Joint modality fusion
        if self.fusion_type == 'concat':
            self.fusion_layer2 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))
        elif self.fusion_type == 'weighted':
            if learnable_alpha:
                self.raw_alpha = nn.Parameter(torch.tensor([float(alpha)]))
            else:
                self.register_buffer('raw_alpha', torch.tensor([float(alpha)]))
            self.joint_proj = nn.Linear(out_channels, out_channels)
        elif self.fusion_type == 'attention':
            self.att_fusion = AttentionFusion(out_channels)
        elif self.fusion_type in ['rna_only', 'aux_only']:
            self.ablation_proj = nn.Linear(out_channels, out_channels)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}. Choose from 'concat', 'weighted', 'attention', 'rna_only', 'aux_only'.")

    def forward(
        self,
        x_sim: torch.Tensor,
        x_dist: torch.Tensor,
        aux_s: torch.Tensor,
        aux_d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_rna: RNA modality embedding
            fused_aux: Auxiliary modality embedding
            fused_pro: Joint multi-omics embedding
        """
        fused_rna = self.fusion_layer1(torch.cat([x_sim, x_dist], dim=1))
        fused_aux = self.fusion_layer_aux(torch.cat([aux_s, aux_d], dim=1))

        if self.fusion_type == 'concat':
            fused_pro = self.fusion_layer2(torch.cat([fused_rna, fused_aux], dim=1))
        elif self.fusion_type == 'weighted':
            alpha = torch.clamp(self.raw_alpha, 0.0, 1.0)
            fused_raw = alpha * fused_rna + (1.0 - alpha) * fused_aux
            fused_pro = self.joint_proj(fused_raw)
        elif self.fusion_type == 'attention':
            fused_pro, _ = self.att_fusion(fused_rna, fused_aux)
        elif self.fusion_type == 'rna_only':
            fused_pro = self.ablation_proj(fused_rna)
        elif self.fusion_type == 'aux_only':
            fused_pro = self.ablation_proj(fused_aux)

        return fused_rna, fused_aux, fused_pro

