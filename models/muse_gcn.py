# ==============================================================================
# models/muse_gcn.py - Unified Muse-GCN Architecture with Configurable Encoders,
#                      Fusion Mechanisms, and Objective Loss Formulation
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from models.backbones import GNNEncoder
from models.fusion import MultiOmicsFusionModule


class MuseGCNCore(nn.Module):
    """
    Core Muse-GCN Module containing:
      - 4 configurable GNN Encoders (RNA Sim, RNA Dist, Aux Sim, Aux Dist)
      - Configurable Fusion Module (Concat, Weighted, Attention, Ablation)
      - Multi-target Reconstruction Decoders
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        q: int,
        gnn_type: str = 'gcn',
        num_layers: int = 1,
        heads: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
        residual: bool = False,
        fusion_type: str = 'concat',
        alpha: float = 0.5,
        learnable_alpha: bool = False
    ):
        super(MuseGCNCore, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # --- 2 RNA Encoders ---
        self.rna_sim = GNNEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            gnn_type=gnn_type,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            residual=residual
        )
        self.rna_dist = GNNEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            gnn_type=gnn_type,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            residual=residual
        )

        # --- 2 Auxiliary Encoders ---
        self.aux_sim = GNNEncoder(
            in_channels=q,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            gnn_type=gnn_type,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            residual=residual
        )
        self.aux_dist = GNNEncoder(
            in_channels=q,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            gnn_type=gnn_type,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            residual=residual
        )

        # --- Multi-Omics Fusion ---
        self.fusion = MultiOmicsFusionModule(
            out_channels=out_channels,
            fusion_type=fusion_type,
            alpha=alpha,
            learnable_alpha=learnable_alpha
        )

        # --- Decoders ---
        self.deconv1 = nn.Linear(out_channels, hidden_channels)
        self.deconv2 = nn.Linear(hidden_channels, in_channels)          # Reconstruct RNA PCA
        self.deconv4 = nn.Linear(hidden_channels, q)                    # Reconstruct Aux
        self.deconv5 = nn.Linear(hidden_channels, q + in_channels)      # Reconstruct Joint

    def forward(
        self,
        x_RNA,
        x_ADT,
        sim_edge_index_rna,
        sim_edge_weight_rna,
        dist_edge_index_rna,
        dist_edge_weight_rna,
        sim_edge_index_aux,
        sim_edge_weight_aux,
        dist_edge_index_aux,
        dist_edge_weight_aux
    ):
        # 1. RNA Stream
        x_sim = self.rna_sim(x_RNA, sim_edge_index_rna, sim_edge_weight_rna)
        x_dist = self.rna_dist(x_RNA, dist_edge_index_rna, dist_edge_weight_rna)

        # 2. Auxiliary Stream
        aux_s = self.aux_sim(x_ADT, sim_edge_index_aux, sim_edge_weight_aux)
        aux_d = self.aux_dist(x_ADT, dist_edge_index_aux, dist_edge_weight_aux)

        # 3. Fusion
        fused_rna, fused_aux, fused_pro = self.fusion(x_sim, x_dist, aux_s, aux_d)

        return x_sim, x_dist, aux_s, aux_d, fused_rna, fused_aux, fused_pro

    def reconstruct(self, z):
        return self.deconv2(F.relu(self.deconv1(z)))

    def reconstruct2(self, z):
        return self.deconv4(F.relu(self.deconv1(z)))

    def reconstruct3(self, z):
        return self.deconv5(F.relu(self.deconv1(z)))


class MuseGCNModel(nn.Module):
    """
    Full Muse-GCN Model with tunable loss weighting:
      Total Loss = beta * L_rec + gamma * L_spatial + delta * L_reg (+ optional kappa * L_aux)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        q: int,
        num_clusters: int,
        beta: float = 25.0,
        gamma: float = 10.0,
        delta: float = 1.0,
        kappa: float = 1.0,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-3,
        gnn_type: str = 'gcn',
        num_layers: int = 1,
        heads: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
        residual: bool = False,
        fusion_type: str = 'concat',
        alpha: float = 0.5,
        learnable_alpha: bool = False
    ):
        super(MuseGCNModel, self).__init__()
        self.gcn = MuseGCNCore(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            q=q,
            gnn_type=gnn_type,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            residual=residual,
            fusion_type=fusion_type,
            alpha=alpha,
            learnable_alpha=learnable_alpha
        )

        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, out_channels))
        self.num_clusters = num_clusters
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.kappa = kappa
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.gcn_input = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.cluster_layer.data)

    def forward(
        self,
        x_RNA,
        x_ADT,
        sim_edge_index_rna,
        sim_edge_weight_rna,
        dist_edge_index_rna,
        dist_edge_weight_rna,
        sim_edge_index_aux,
        sim_edge_weight_aux,
        dist_edge_index_aux,
        dist_edge_weight_aux
    ):
        return self.gcn(
            x_RNA, x_ADT,
            sim_edge_index_rna, sim_edge_weight_rna,
            dist_edge_index_rna, dist_edge_weight_rna,
            sim_edge_index_aux, sim_edge_weight_aux,
            dist_edge_index_aux, dist_edge_weight_aux
        )

    def compute_regularization_loss(self):
        l1_loss = sum(torch.sum(torch.abs(p)) for p in self.parameters())
        l2_loss = sum(torch.sum(p ** 2) for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

    def cosine_similarity(self, emb):
        mat = torch.matmul(emb, emb.T)
        norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
        mat = torch.div(mat, torch.matmul(norm, norm.T))
        mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
        mat = mat - torch.diag_embed(torch.diag(mat))
        return mat

    def spatial_regularization_loss(self, emb, dist_edge_index, dist_edge_weight):
        num_nodes = emb.size(0)
        graph_nei = torch.sparse_coo_tensor(
            dist_edge_index,
            torch.ones_like(dist_edge_weight),
            size=(num_nodes, num_nodes)
        ).to_dense()
        graph_neg = 1 - graph_nei
        sim_mat = torch.sigmoid(self.cosine_similarity(emb))

        neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
        neg_loss = torch.mul(graph_neg, torch.log(1 - sim_mat + 1e-10)).mean()

        return -(neigh_loss + neg_loss) / 2

    def compute_losses(
        self,
        x_RNA,
        x_ADT,
        x_sim,
        x_dist,
        aux_s,
        aux_d,
        fused_rna,
        fused_aux,
        fused_pro,
        combined_raw
    ):
        l_rec = F.mse_loss(combined_raw, self.gcn.reconstruct3(fused_pro))
        l_sim = F.mse_loss(x_RNA, self.gcn.reconstruct(x_sim))
        l_dist = F.mse_loss(x_RNA, self.gcn.reconstruct(x_dist))
        l_aux_s = F.mse_loss(x_ADT, self.gcn.reconstruct2(aux_s))
        l_aux_d = F.mse_loss(x_ADT, self.gcn.reconstruct2(aux_d))
        l_rec_total = l_rec + l_sim + l_dist + l_aux_s + l_aux_d

        l_spatial = self.spatial_regularization_loss(
            fused_rna,
            dist_edge_index=self.gcn_input.dist_edge_index_rna,
            dist_edge_weight=self.gcn_input.dist_edge_weight_rna
        )

        reg_loss = self.compute_regularization_loss()

        recon_component = self.beta * l_rec_total
        spatial_component = self.gamma * l_spatial
        reg_component = self.delta * reg_loss

        total_loss = recon_component + spatial_component + reg_component

        return total_loss, recon_component, spatial_component, reg_component

