# ==============================================================================
# trainer.py - Flexible Muse-GCN Training & Evaluation Engine
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any, Optional

try:
    import optuna
except ImportError:
    optuna = None

from clustering.cluster_engine import cluster_embeddings
from utils.metrics import compute_all_metrics


def evaluate_embeddings(model: nn.Module, data) -> Dict[str, np.ndarray]:
    """
    Extract multi-omics embeddings from model.
    """
    model.eval()
    with torch.no_grad():
        x_sim, x_dist, aux_s, aux_d, fused_rna, fused_aux, fused_pro = model(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index_rna, data.sim_edge_weight_rna,
            data.dist_edge_index_rna, data.dist_edge_weight_rna,
            data.sim_edge_index_aux, data.sim_edge_weight_aux,
            data.dist_edge_index_aux, data.dist_edge_weight_aux
        )
    return {
        'fused_pro': fused_pro.cpu().numpy(),
        'fused_rna': fused_rna.cpu().numpy(),
        'fused_aux': fused_aux.cpu().numpy(),
        'x_sim': x_sim.cpu().numpy(),
        'x_dist': x_dist.cpu().numpy(),
        'aux_s': aux_s.cpu().numpy(),
        'aux_d': aux_d.cpu().numpy()
    }


def train_muse_gcn(
    model: nn.Module,
    data,
    epochs: int = 350,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = 'adam',
    num_clusters: int = 10,
    clustering_method: str = 'kmeans',
    cluster_restarts: int = 10,
    true_labels: Optional[np.ndarray] = None,
    verbose: bool = True,
    eval_every_epoch: bool = False,
    trial: Optional[Any] = None  # Optuna trial for pruning
) -> Dict[str, Any]:
    """
    Comprehensive training loop supporting:
      - Optimizer choice: 'adam', 'adamw'
      - Weight decay
      - Real-time Silhouette and ARI tracking
      - Optuna pruning hooks
      - Modular clustering methods
    """
    opt_name = optimizer_name.lower()
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    combined_raw = torch.cat([data.x_RNA, data.x_ADT], dim=1)

    loss_history = []
    recon_loss_history = []
    spatial_loss_history = []
    reg_loss_history = []
    epoch_sil_history = []
    epoch_ari_history = []

    best_score = -1.0
    best_embeddings = None
    best_labels = None
    best_epoch = 0

    pbar = tqdm(range(epochs), desc="Training Muse-GCN", disable=not verbose)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        x_sim, x_dist, aux_s, aux_d, fused_rna, fused_aux, fused_pro = model(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index_rna, data.sim_edge_weight_rna,
            data.dist_edge_index_rna, data.dist_edge_weight_rna,
            data.sim_edge_index_aux, data.sim_edge_weight_aux,
            data.dist_edge_index_aux, data.dist_edge_weight_aux
        )

        model.gcn_input = data
        loss, l_rec_val, l_spatial_val, reg_loss_val = model.compute_losses(
            data.x_RNA, data.x_ADT,
            x_sim, x_dist, aux_s, aux_d,
            fused_rna, fused_aux, fused_pro,
            combined_raw
        )

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        recon_loss_history.append(l_rec_val.item())
        spatial_loss_history.append(l_spatial_val.item())
        reg_loss_history.append(reg_loss_val.item())

        is_last_epoch = (epoch == epochs - 1)
        do_eval = eval_every_epoch or is_last_epoch or (trial is not None and (epoch + 1) % 25 == 0)

        if do_eval:
            embs = evaluate_embeddings(model, data)['fused_pro']
            pred_labels = cluster_embeddings(
                embs,
                num_clusters=num_clusters,
                method=clustering_method,
                n_init=cluster_restarts,
                random_state=42
            )
            metrics = compute_all_metrics(embs, pred_labels, true_labels)
            sil = metrics['Silhouette']
            ari = metrics['ARI']
            epoch_sil_history.append(sil)
            epoch_ari_history.append(ari)

            if sil > best_score:
                best_score = sil
                best_embeddings = embs.copy()
                best_labels = pred_labels.copy()
                best_epoch = epoch + 1

            # Optuna Pruning Hook
            if trial is not None and (epoch + 1) % 25 == 0:
                trial.report(sil, epoch)
                if trial.should_prune():
                    if optuna is not None:
                        raise optuna.TrialPruned()

            postfix = {'loss': f'{loss.item():.4f}', 'sil': f'{sil:.4f}', 'ari': f'{ari:.4f}'}
            pbar.set_postfix(postfix)

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or is_last_epoch):
                tqdm.write(
                    f"Epoch {epoch + 1:4d}/{epochs} | Loss: {loss.item():.4f} | "
                    f"Recon: {l_rec_val.item():.4f} | Sil: {sil:.4f} | ARI: {ari:.4f}"
                )
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Final evaluation on last epoch
    final_embs = evaluate_embeddings(model, data)['fused_pro']
    final_labels = cluster_embeddings(
        final_embs,
        num_clusters=num_clusters,
        method=clustering_method,
        n_init=cluster_restarts,
        random_state=42
    )
    final_metrics = compute_all_metrics(final_embs, final_labels, true_labels)

    return {
        'model': model,
        'final_embeddings': final_embs,
        'final_labels': final_labels,
        'final_sil': final_metrics['Silhouette'],
        'final_ari': final_metrics['ARI'],
        'final_nmi': final_metrics['NMI'],
        'final_ami': final_metrics['AMI'],
        'final_chi': final_metrics['CHI'],
        'final_dbi': final_metrics['DBI'],
        'final_homo': final_metrics['Homogeneity'],
        'final_v_meas': final_metrics['V-measure'],
        'best_score': best_score,
        'best_epoch': best_epoch,
        'loss_history': loss_history,
        'recon_loss_history': recon_loss_history,
        'spatial_loss_history': spatial_loss_history,
        'reg_loss_history': reg_loss_history,
        'epoch_sil_history': epoch_sil_history,
        'epoch_ari_history': epoch_ari_history
    }

