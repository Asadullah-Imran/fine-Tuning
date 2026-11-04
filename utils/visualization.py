# ==============================================================================
# utils/visualization.py - Training Curves, Spatial Maps, UMAPs, Violins
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.metrics import silhouette_samples
from typing import Dict, Any, Optional


def plot_training_curves(training_results: Dict[str, Any], dataset_name: str = "Dataset", save_path: Optional[str] = None, show: bool = False):
    """
    Plot Loss Curve, Silhouette Score Curve, and ARI Curve across training epochs.
    """
    epochs_range = range(1, len(training_results['loss_history']) + 1)
    has_ari = len(training_results.get('epoch_ari_history', [])) > 0

    fig, axes = plt.subplots(1, 3 if has_ari else 2, figsize=(18 if has_ari else 12, 4.5), dpi=150)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 1. Loss Curve
    axes[0].plot(epochs_range, training_results['loss_history'], color='#1f77b4', linewidth=2.0, label='Total Loss')
    if 'recon_loss_history' in training_results and len(training_results['recon_loss_history']) == len(epochs_range):
        axes[0].plot(epochs_range, training_results['recon_loss_history'], color='#aec7e8', linestyle=':', label='Recon Loss')
    axes[0].set_title(f'Loss Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True)

    # 2. Silhouette Score Curve
    sil_hist = training_results.get('epoch_sil_history', [])
    if len(sil_hist) == len(epochs_range):
        axes[1].plot(epochs_range, sil_hist, color='#2ca02c', linewidth=2.0, label='Silhouette Score')
    else:
        x_vals = [len(epochs_range)] if len(sil_hist) == 1 else np.linspace(1, len(epochs_range), len(sil_hist))
        axes[1].plot(x_vals, sil_hist, marker='o', color='#2ca02c', linewidth=2.0, label='Silhouette Score')
    last_sil = training_results.get('final_sil', training_results.get('last_sil', 0.0))
    axes[1].scatter([len(epochs_range)], [last_sil], color='#2ca02c', s=50, zorder=5, label=f'Final Sil: {last_sil:.4f}')
    axes[1].set_title(f'Silhouette Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(frameon=True, loc='lower right')

    # 3. ARI Curve
    if has_ari:
        ari_hist = training_results.get('epoch_ari_history', [])
        if len(ari_hist) == len(epochs_range):
            axes[2].plot(epochs_range, ari_hist, color='#ff7f0e', linewidth=2.0, label='Epoch ARI')
        else:
            x_vals = [len(epochs_range)] if len(ari_hist) == 1 else np.linspace(1, len(epochs_range), len(ari_hist))
            axes[2].plot(x_vals, ari_hist, marker='o', color='#ff7f0e', linewidth=2.0, label='Epoch ARI')
        last_ari = training_results.get('final_ari', training_results.get('last_ari', 0.0))
        axes[2].scatter([len(epochs_range)], [last_ari], color='#ff7f0e', s=50, zorder=5, label=f'Final ARI: {last_ari:.4f}')
        axes[2].set_title(f'ARI Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('ARI', fontsize=11)
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].legend(frameon=True, loc='lower right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_visualizations(
    adata_RNA,
    results: Dict[str, Any],
    dataset_name: str = "Dataset",
    seed: int = 42,
    rna_pca_comps: int = 60,
    true_labels: Optional[np.ndarray] = None,
    output_dir: str = "results",
    show: bool = False
):
    """
    Generate and save complete publication visualizations:
    1. Training Curves (Loss, Silhouette, ARI)
    2. Ground Truth vs Predicted Spatial Domains
    3. UMAP Scatter Plots (Ground Truth & Predicted Domains)
    4. Violin Plots (Silhouette Coefficient & Latent Dim Profiles)
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(os.path.join(plots_dir, "curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "spatial"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "umap"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "violin"), exist_ok=True)

    final_embeddings = results.get('final_embeddings')
    final_labels = results.get('final_labels')
    sil = results.get('final_sil', 0.0)
    ari = results.get('final_ari', 0.0)

    if true_labels is not None:
        adata_RNA.obs['ground_truth'] = pd.Categorical(true_labels)
    elif 'ground_truth' not in adata_RNA.obs:
        adata_RNA.obs['ground_truth'] = pd.Categorical(final_labels.astype(str))

    adata_RNA.obsm['Muse-GCN_Embedding'] = final_embeddings
    adata_RNA.obs['predicted_domain'] = pd.Categorical(final_labels.astype(str))

    # 1. Training Curves
    curve_path = os.path.join(plots_dir, "curves", f"{dataset_name}_seed{seed}_training_curves.png")
    plot_training_curves(
        results,
        dataset_name=f"{dataset_name} (RNA PCA {rna_pca_comps}, Seed {seed})",
        save_path=curve_path,
        show=show
    )

    # 2. Spatial Domains Plot
    spatial_coords = adata_RNA.obsm.get('spatial', None)
    if spatial_coords is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc.pl.embedding(
            adata_RNA,
            basis='spatial',
            color='ground_truth',
            ax=axes[0],
            show=False,
            title='Ground Truth',
            frameon=True
        )
        axes[0].set_xlabel("spatial1")
        axes[0].set_ylabel("spatial2")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        sc.pl.embedding(
            adata_RNA,
            basis='spatial',
            color='predicted_domain',
            ax=axes[1],
            show=False,
            title='Muse-GCN Predicted',
            frameon=True
        )
        axes[1].set_xlabel("spatial1")
        axes[1].set_ylabel("spatial2")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        plt.suptitle(f"Spatial Domains - {dataset_name} (ARI: {ari:.4f})", fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        spatial_path = os.path.join(plots_dir, "spatial", f"{dataset_name}_seed{seed}_spatial.png")
        plt.savefig(spatial_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    # 3. UMAP Plot
    sc.pp.neighbors(adata_RNA, use_rep='Muse-GCN_Embedding')
    sc.tl.umap(adata_RNA)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(
        adata_RNA,
        color='ground_truth',
        ax=axes[0],
        show=False,
        title='UMAP (Ground Truth)'
    )
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    sc.pl.umap(
        adata_RNA,
        color='predicted_domain',
        ax=axes[1],
        show=False,
        title='UMAP (Muse-GCN)'
    )
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.suptitle(f"UMAP - {dataset_name} (Sil: {sil:.4f})", fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    umap_path = os.path.join(plots_dir, "umap", f"{dataset_name}_seed{seed}_umap.png")
    plt.savefig(umap_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    # 4. Violin Plots
    sample_sil_values = silhouette_samples(final_embeddings, final_labels)
    adata_RNA.obs['silhouette_coefficient'] = sample_sil_values
    adata_RNA.obs['Latent_Dim_1'] = final_embeddings[:, 0]
    adata_RNA.obs['Latent_Dim_2'] = final_embeddings[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    sns.violinplot(
        data=adata_RNA.obs,
        x='predicted_domain',
        y='silhouette_coefficient',
        palette='Set2',
        inner='quartile',
        ax=axes[0]
    )
    axes[0].axhline(sil, color='red', linestyle='--', label=f'Mean Sil: {sil:.4f}')
    axes[0].set_title("Silhouette Coefficient per Domain", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Domain")
    axes[0].set_ylabel("Silhouette Coefficient")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    sns.violinplot(
        data=adata_RNA.obs,
        x='predicted_domain',
        y='Latent_Dim_1',
        palette='tab10',
        inner='box',
        ax=axes[1]
    )
    axes[1].set_title("Latent Dimension 1 per Domain", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Domain")
    axes[1].set_ylabel("Latent Embedding Dim 1")
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.suptitle(f"Violin Profiles - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    violin_path = os.path.join(plots_dir, "violin", f"{dataset_name}_seed{seed}_violin.png")
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

