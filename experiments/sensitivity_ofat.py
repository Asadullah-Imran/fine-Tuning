# ==============================================================================
# experiments/sensitivity_ofat.py - One-Factor-At-A-Time (OFAT) Sensitivity Runner
# ==============================================================================

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch

from utils.data_loader import load_dataset, set_seed
from graphs.builder import build_4encoder_graphs
from models.muse_gcn import MuseGCNModel
from trainer import train_muse_gcn

OFAT_GRIDS = {
    'beta': [5.0, 10.0, 15.0, 25.0, 50.0, 100.0],
    'gamma': [0.1, 1.0, 5.0, 10.0, 20.0, 30.0],
    'delta': [0.01, 0.1, 1.0, 5.0, 10.0],
    'lr': [0.0001, 0.0005, 0.001, 0.005],
    'spatial_k': [4, 6, 8, 10, 15, 20]
}

BASELINE_DEFAULTS = {
    'beta': 25.0,
    'gamma': 10.0,
    'delta': 1.0,
    'lr': 0.001,
    'spatial_k': 15,
    'hidden_dim': 512,
    'out_dim': 64,
    'dropout': 0.0,
    'epochs': 250
}


def run_ofat_sweep(
    param_name: str,
    dataset_name: str = "10x_human_lymph_node_A1",
    values: list = None,
    seeds: list = [42, 1234, 2024],
    data_dir: str = "data",
    output_dir: str = "results/sensitivity",
    device: str = None
):
    """
    Run OFAT parameter sweep: varying only param_name while freezing all other parameters.
    """
    if values is None:
        values = OFAT_GRIDS.get(param_name, [1, 2, 5, 10])

    dev_str = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(dev_str)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[OFAT Sensitivity] Loading dataset: {dataset_name}...")
    data_pack = load_dataset(dataset_name, data_dir=data_dir, rna_pca_comps=60)

    sweep_results = []

    print("\n" + "=" * 75)
    print(f" 📈 RUNNING OFAT SENSITIVITY SWEEP: {param_name.upper()} ".center(75, "="))
    print("=" * 75)
    print(f"• Parameter   : {param_name}")
    print(f"• Values      : {values}")
    print(f"• Seeds       : {seeds}")
    print("=" * 75 + "\n")

    for val in values:
        cfg = BASELINE_DEFAULTS.copy()
        cfg[param_name] = val

        for seed in seeds:
            set_seed(seed)

            # Rebuild graph if graph parameter is varied
            data = build_4encoder_graphs(
                RNA_expression=data_pack['RNA_expr'],
                ADT_expression=data_pack['ADT_expr'],
                cell_positions=data_pack['cell_positions'],
                device=dev,
                spatial_k=int(cfg['spatial_k']),
                feature_k_rna=int(cfg['spatial_k']),
                feature_k_aux=int(cfg['spatial_k'])
            )

            model = MuseGCNModel(
                in_channels=data.x_RNA.shape[1],
                hidden_channels=cfg['hidden_dim'],
                out_channels=cfg['out_dim'],
                q=data.x_ADT.shape[1],
                num_clusters=data_pack['num_clusters'],
                beta=cfg['beta'],
                gamma=cfg['gamma'],
                delta=cfg['delta'],
                dropout=cfg['dropout']
            ).to(dev)

            res = train_muse_gcn(
                model=model,
                data=data,
                epochs=cfg['epochs'],
                lr=cfg['lr'],
                num_clusters=data_pack['num_clusters'],
                true_labels=data_pack['true_labels'],
                verbose=False
            )

            record = {
                'parameter': param_name,
                'value': val,
                'seed': seed,
                'ARI': res['final_ari'],
                'NMI': res['final_nmi'],
                'Silhouette': res['final_sil'],
                'AMI': res['final_ami']
            }
            sweep_results.append(record)
            print(f"  • {param_name} = {val:<8} | Seed: {seed} -> ARI: {res['final_ari']:.4f} | Sil: {res['final_sil']:.4f}")

    df_sweep = pd.DataFrame(sweep_results)
    csv_file = os.path.join(output_dir, f"ofat_{param_name}_{dataset_name}.csv")
    df_sweep.to_csv(csv_file, index=False)
    print(f"\n💾 Saved OFAT sweep CSV to: {csv_file}")

    # Plot Sensitivity Curve
    agg_df = df_sweep.groupby('value').agg({'ARI': ['mean', 'std'], 'Silhouette': ['mean', 'std']}).reset_index()
    agg_df.columns = ['value', 'ARI_mean', 'ARI_std', 'Sil_mean', 'Sil_std']

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=150)
    color = '#1f77b4'
    ax1.set_xlabel(f"{param_name.upper()} Value", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Adjusted Rand Index (ARI)", color=color, fontsize=12, fontweight='bold')
    ax1.errorbar(
        agg_df['value'], agg_df['ARI_mean'], yerr=agg_df['ARI_std'],
        fmt='-o', color=color, capsize=5, linewidth=2, label='ARI'
    )
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color2 = '#2ca02c'
    ax2.set_ylabel("Silhouette Score", color=color2, fontsize=12, fontweight='bold')
    ax2.errorbar(
        agg_df['value'], agg_df['Sil_mean'], yerr=agg_df['Sil_std'],
        fmt='--s', color=color2, capsize=5, linewidth=2, label='Silhouette'
    )
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f"Sensitivity Analysis: {param_name.upper()} ({dataset_name})", fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    plot_file = os.path.join(output_dir, f"ofat_{param_name}_{dataset_name}_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved Sensitivity Plot to: {plot_file}\n")

    return df_sweep


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OFAT Sensitivity Runner for Muse-GCN")
    parser.add_argument('--param', type=str, default='beta', choices=['beta', 'gamma', 'delta', 'lr', 'spatial_k', 'all'])
    parser.add_argument('--dataset', type=str, default='10x_human_lymph_node_A1')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1234, 2024])
    parser.add_argument('--output_dir', type=str, default='results/sensitivity')

    args = parser.parse_args()

    if args.param == 'all':
        for p in ['beta', 'gamma', 'delta', 'lr', 'spatial_k']:
            run_ofat_sweep(p, dataset_name=args.dataset, seeds=args.seeds, output_dir=args.output_dir)
    else:
        run_ofat_sweep(args.param, dataset_name=args.dataset, seeds=args.seeds, output_dir=args.output_dir)

