# ==============================================================================
# experiments/ablation_runner.py - Systematic Architectural & Component Ablation
# ==============================================================================

import os
import argparse
import pandas as pd
import torch

from utils.data_loader import load_dataset, set_seed
from graphs.builder import build_4encoder_graphs
from models.muse_gcn import MuseGCNModel
from trainer import train_muse_gcn


def run_gnn_backbone_ablation(dataset_name: str = "10x_human_lymph_node_A1", seeds: list = [42, 1234, 2024], output_dir: str = "results/ablations"):
    """Compare GCN vs GAT vs GraphSAGE."""
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_pack = load_dataset(dataset_name, rna_pca_comps=60)
    data = build_4encoder_graphs(data_pack['RNA_expr'], data_pack['ADT_expr'], data_pack['cell_positions'], device=dev)

    records = []
    backbones = ['gcn', 'gat', 'sage']

    print("\n" + "=" * 70)
    print(" 🔬 EXPERIMENT C: GNN BACKBONE BENCHMARK ".center(70, "="))
    print("=" * 70)

    for gnn in backbones:
        for seed in seeds:
            set_seed(seed)
            model = MuseGCNModel(
                in_channels=data.x_RNA.shape[1],
                hidden_channels=512,
                out_channels=64,
                q=data.x_ADT.shape[1],
                num_clusters=data_pack['num_clusters'],
                gnn_type=gnn,
                num_layers=1
            ).to(dev)

            res = train_muse_gcn(model, data, epochs=250, num_clusters=data_pack['num_clusters'], true_labels=data_pack['true_labels'], verbose=False)
            records.append({'GNN_Backbone': gnn.upper(), 'Seed': seed, 'ARI': res['final_ari'], 'Silhouette': res['final_sil'], 'NMI': res['final_nmi']})
            print(f"  • Backbone: {gnn.upper():<6} | Seed: {seed} -> ARI: {res['final_ari']:.4f} | Sil: {res['final_sil']:.4f}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, f"ablation_gnn_backbone_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    summary = df.groupby('GNN_Backbone')[['ARI', 'Silhouette', 'NMI']].agg(['mean', 'std'])
    print("\n📋 GNN BACKBONE SUMMARY:\n", summary)
    return df


def run_depth_ablation(dataset_name: str = "10x_human_lymph_node_A1", seeds: list = [42, 1234, 2024], output_dir: str = "results/ablations"):
    """Compare 1-layer vs 2-layer vs 3-layer GCN."""
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_pack = load_dataset(dataset_name, rna_pca_comps=60)
    data = build_4encoder_graphs(data_pack['RNA_expr'], data_pack['ADT_expr'], data_pack['cell_positions'], device=dev)

    records = []
    depths = [1, 2, 3]

    print("\n" + "=" * 70)
    print(" 🔬 EXPERIMENT D: ENCODER DEPTH (LAYERS) ABLATION ".center(70, "="))
    print("=" * 70)

    for d in depths:
        for seed in seeds:
            set_seed(seed)
            model = MuseGCNModel(
                in_channels=data.x_RNA.shape[1],
                hidden_channels=512,
                out_channels=64,
                q=data.x_ADT.shape[1],
                num_clusters=data_pack['num_clusters'],
                gnn_type='gcn',
                num_layers=d
            ).to(dev)

            res = train_muse_gcn(model, data, epochs=250, num_clusters=data_pack['num_clusters'], true_labels=data_pack['true_labels'], verbose=False)
            records.append({'Layers': d, 'Seed': seed, 'ARI': res['final_ari'], 'Silhouette': res['final_sil'], 'NMI': res['final_nmi']})
            print(f"  • Layers: {d} | Seed: {seed} -> ARI: {res['final_ari']:.4f} | Sil: {res['final_sil']:.4f}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, f"ablation_encoder_depth_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    summary = df.groupby('Layers')[['ARI', 'Silhouette', 'NMI']].agg(['mean', 'std'])
    print("\n📋 ENCODER DEPTH SUMMARY:\n", summary)
    return df


def run_fusion_ablation(dataset_name: str = "10x_human_lymph_node_A1", seeds: list = [42, 1234, 2024], output_dir: str = "results/ablations"):
    """Compare RNA-only, Aux-only, Concat, Weighted (alpha=0.5), Attention Fusion."""
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_pack = load_dataset(dataset_name, rna_pca_comps=60)
    data = build_4encoder_graphs(data_pack['RNA_expr'], data_pack['ADT_expr'], data_pack['cell_positions'], device=dev)

    records = []
    fusion_configs = [
        ('rna_only', 0.5, 'RNA Only'),
        ('aux_only', 0.5, 'Aux Omics Only'),
        ('concat', 0.5, 'Linear Concat (Baseline)'),
        ('weighted', 0.5, 'Weighted Fusion (α=0.5)'),
        ('weighted', 0.7, 'Weighted Fusion (α=0.7)'),
        ('attention', 0.5, 'Attention Fusion')
    ]

    print("\n" + "=" * 70)
    print(" 🔬 EXPERIMENT B: MODALITY FUSION & CONTRIBUTION ABLATION ".center(70, "="))
    print("=" * 70)

    for f_type, alpha, label in fusion_configs:
        for seed in seeds:
            set_seed(seed)
            model = MuseGCNModel(
                in_channels=data.x_RNA.shape[1],
                hidden_channels=512,
                out_channels=64,
                q=data.x_ADT.shape[1],
                num_clusters=data_pack['num_clusters'],
                fusion_type=f_type,
                alpha=alpha
            ).to(dev)

            res = train_muse_gcn(model, data, epochs=250, num_clusters=data_pack['num_clusters'], true_labels=data_pack['true_labels'], verbose=False)
            records.append({'Configuration': label, 'Fusion_Type': f_type, 'Alpha': alpha, 'Seed': seed, 'ARI': res['final_ari'], 'Silhouette': res['final_sil']})
            print(f"  • {label:<28} | Seed: {seed} -> ARI: {res['final_ari']:.4f} | Sil: {res['final_sil']:.4f}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, f"ablation_fusion_strategies_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    summary = df.groupby('Configuration')[['ARI', 'Silhouette']].agg(['mean', 'std'])
    print("\n📋 FUSION STRATEGIES SUMMARY:\n", summary)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ablation Suite for Muse-GCN")
    parser.add_argument('--experiment', type=str, default='fusion', choices=['backbone', 'depth', 'fusion', 'all'])
    parser.add_argument('--dataset', type=str, default='10x_human_lymph_node_A1')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1234, 2024])
    parser.add_argument('--output_dir', type=str, default='results/ablations')

    args = parser.parse_args()

    if args.experiment in ['backbone', 'all']:
        run_gnn_backbone_ablation(args.dataset, args.seeds, args.output_dir)
    if args.experiment in ['depth', 'all']:
        run_depth_ablation(args.dataset, args.seeds, args.output_dir)
    if args.experiment in ['fusion', 'all']:
        run_fusion_ablation(args.dataset, args.seeds, args.output_dir)

