# ==============================================================================
# train.py - Unified Entrypoint for Muse-GCN Training, Tuning & Ablations
# ==============================================================================

import os
import argparse
import pandas as pd
import torch
from typing import List, Union, Optional

from utils.data_loader import (
    DATASET_REGISTRY,
    DATASET_LIST,
    DEFAULT_SEEDS,
    set_seed,
    load_dataset
)
from utils.visualization import plot_all_visualizations
from utils.dashboard_exporter import export_dashboard_experiment
from graphs.builder import build_4encoder_graphs
from models.muse_gcn import MuseGCNModel
from trainer import train_muse_gcn


def run_experiment(
    datasets: Union[str, int, List[Union[str, int]]] = "all",
    seeds: Optional[List[int]] = None,
    n_seeds: Optional[int] = None,
    rna_pca_comps: int = 60,
    epochs: int = 350,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
    beta: float = 25.0,
    gamma: float = 10.0,
    delta: float = 1.0,
    spatial_k: int = 15,
    feature_k_rna: Optional[int] = None,
    feature_k_aux: Optional[int] = None,
    hidden_dim: int = 512,
    out_dim: int = 64,
    dropout: float = 0.0,
    gnn_type: str = "gcn",
    num_layers: int = 1,
    fusion_type: str = "concat",
    alpha: float = 0.5,
    clustering_method: str = "kmeans",
    cluster_restarts: int = 10,
    device: Optional[str] = None,
    visualize: bool = True,
    show_plots: bool = False,
    output_dir: str = "results",
    data_dir: str = "data",
    api_url: Optional[str] = None,
    eval_every_epoch: bool = False
):
    """
    Run Muse-GCN training and evaluation across specified datasets and seeds.
    """
    dev_str = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(dev_str)

    # Resolve datasets
    if datasets == "all" or datasets is None:
        dataset_names = DATASET_LIST
    elif isinstance(datasets, (str, int)):
        dataset_names = [DATASET_LIST[datasets]] if isinstance(datasets, int) else [datasets]
    else:
        dataset_names = [DATASET_LIST[d] if isinstance(d, int) else str(d) for d in datasets]

    # Resolve seeds
    if seeds is None or seeds == "all":
        run_seeds = DEFAULT_SEEDS
    else:
        run_seeds = [int(s) for s in seeds]

    if n_seeds is not None and n_seeds > 0:
        run_seeds = run_seeds[:n_seeds]

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print("\n" + "=" * 80)
    print(" 🚀 STARTING MODULAR MUSE-GCN EXPERIMENT ".center(80, "="))
    print("=" * 80)
    print(f"• Datasets      : {dataset_names}")
    print(f"• Seeds ({len(run_seeds):2d})   : {run_seeds}")
    print(f"• Architecture  : {gnn_type.upper()} ({num_layers} layer{'s' if num_layers > 1 else ''}) | Fusion: {fusion_type.upper()}")
    print(f"• Graph k-NN    : Spatial={spatial_k}, RNA={feature_k_rna or spatial_k}, Aux={feature_k_aux or spatial_k}")
    print(f"• Hyperparams   : Epochs={epochs}, LR={lr}, Beta={beta}, Gamma={gamma}, Delta={delta}")
    print(f"• Dimensions    : Hidden={hidden_dim}, Out={out_dim} | Device: {dev_str}")
    print("=" * 80 + "\n")

    for dataset_name in dataset_names:
        print("\n" + "#" * 80)
        print(f" DATASET: {dataset_name} ".center(80, "#"))
        print("#" * 80)

        data_pack = load_dataset(dataset_name, data_dir=data_dir, rna_pca_comps=rna_pca_comps)
        adata_RNA = data_pack['adata_RNA']
        true_labels = data_pack['true_labels']
        num_clusters = data_pack['num_clusters']

        # Build Multi-Omics Graph
        data = build_4encoder_graphs(
            RNA_expression=data_pack['RNA_expr'],
            ADT_expression=data_pack['ADT_expr'],
            cell_positions=data_pack['cell_positions'],
            device=dev,
            spatial_k=spatial_k,
            feature_k_rna=feature_k_rna,
            feature_k_aux=feature_k_aux
        )

        dataset_results = []

        for seed in run_seeds:
            print(f"\n" + "-" * 60)
            print(f" Dataset: {dataset_name} | Seed: {seed} ".center(60, "-"))
            print("-" * 60)

            set_seed(seed)

            model = MuseGCNModel(
                in_channels=data.x_RNA.shape[1],
                hidden_channels=hidden_dim,
                out_channels=out_dim,
                q=data.x_ADT.shape[1],
                num_clusters=num_clusters,
                beta=beta,
                gamma=gamma,
                delta=delta,
                gnn_type=gnn_type,
                num_layers=num_layers,
                dropout=dropout,
                fusion_type=fusion_type,
                alpha=alpha
            ).to(dev)

            result = train_muse_gcn(
                model=model,
                data=data,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                optimizer_name=optimizer_name,
                num_clusters=num_clusters,
                clustering_method=clustering_method,
                cluster_restarts=cluster_restarts,
                true_labels=true_labels,
                verbose=True,
                eval_every_epoch=eval_every_epoch
            )

            print(f"\nSummary for {dataset_name} | Seed: {seed}")
            print(f" • Silhouette : {result['final_sil']:.4f} | ARI: {result['final_ari']:.4f} | NMI: {result['final_nmi']:.4f}")
            print(f" • CHI: {result['final_chi']:.2f} | DBI: {result['final_dbi']:.4f} | V-measure: {result['final_v_meas']:.4f}")

            if visualize:
                plot_all_visualizations(
                    adata_RNA=adata_RNA,
                    results=result,
                    dataset_name=dataset_name,
                    seed=seed,
                    rna_pca_comps=rna_pca_comps,
                    true_labels=true_labels,
                    output_dir=output_dir,
                    show=show_plots
                )

            # Export Standardized JSON artifacts for Dashboard
            export_dashboard_experiment(
                model_id=f"MUSE-{gnn_type.upper()}-{fusion_type}",
                model_name=f"Muse-GCN ({gnn_type.upper()}, {fusion_type})",
                dataset_name=dataset_name,
                seed=seed,
                metrics_dict={
                    "ARI": result['final_ari'],
                    "NMI": result['final_nmi'],
                    "Silhouette": result['final_sil'],
                    "AMI": result['final_ami'],
                    "CHI": result['final_chi'],
                    "DBI": result['final_dbi'],
                    "Homogeneity": result['final_homo'],
                    "V-measure": result['final_v_meas']
                },
                training_results=result,
                hyperparameters={
                    "epochs": epochs,
                    "lr": lr,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": delta,
                    "spatial_k": spatial_k,
                    "gnn_type": gnn_type,
                    "num_layers": num_layers,
                    "fusion_type": fusion_type,
                    "alpha": alpha,
                    "hidden_dim": hidden_dim,
                    "out_dim": out_dim,
                    "rna_pca_comps": rna_pca_comps
                },
                output_dir=output_dir,
                api_url=api_url
            )

            record = {
                'dataset': dataset_name,
                'seed': seed,
                'Silhouette': result['final_sil'],
                'ARI': result['final_ari'],
                'NMI': result['final_nmi'],
                'AMI': result['final_ami'],
                'CHI': result['final_chi'],
                'DBI': result['final_dbi'],
                'Homogeneity': result['final_homo'],
                'V-measure': result['final_v_meas']
            }
            dataset_results.append(record)
            all_results.append(record)

        df_ds = pd.DataFrame(dataset_results)
        df_ds.to_csv(os.path.join(output_dir, f"MuseGCN_{dataset_name}_results.csv"), index=False)

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, "MuseGCN_all_results.csv"), index=False)

    summary_cols = ['ARI', 'Silhouette', 'NMI', 'AMI', 'CHI', 'DBI']
    print("\n" + "=" * 88)
    print(" ALL EXPERIMENTS COMPLETED - DATASET SUMMARY ".center(88, "="))
    print("=" * 88)
    print(df_all.groupby('dataset')[summary_cols].mean().to_string())
    print("-" * 88)
    print(" OVERALL MEAN ACROSS ALL DATASETS & SEEDS ".center(88, "-"))
    print(df_all[summary_cols].mean().to_frame().T.to_string(index=False))
    print("=" * 88 + "\n")

    return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Unified Muse-GCN Training & Optimization CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Operation Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'ofat', 'hpo', 'ablation'],
                        help="Execution mode: standard training, OFAT sensitivity sweep, Optuna HPO, or Ablation study")

    # Dataset & Seed Config
    parser.add_argument('--datasets', nargs='+', default=['all'], help="Dataset indices, names, or 'all'")
    parser.add_argument('--seeds', nargs='+', default=None, help="List of seeds (e.g. 42 1234 2024) or 'all'")
    parser.add_argument('--n_seeds', type=int, default=None, help="Limit to first N seeds")
    parser.add_argument('--rna_pca_comps', type=int, default=60, help="Number of RNA PCA components")

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=350, help="Training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help="Optimizer")
    parser.add_argument('--beta', type=float, default=25.0, help="Reconstruction loss weight")
    parser.add_argument('--gamma', type=float, default=10.0, help="Spatial regularization weight")
    parser.add_argument('--delta', type=float, default=1.0, help="Regularization loss weight")

    # Graph & Architecture
    parser.add_argument('--spatial_k', type=int, default=15, help="Spatial KNN neighbors")
    parser.add_argument('--feature_k_rna', type=int, default=None, help="RNA feature KNN neighbors")
    parser.add_argument('--feature_k_aux', type=int, default=None, help="Aux feature KNN neighbors")
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'sage', 'gin', 'gatv2'], help="GNN Backbone")
    parser.add_argument('--num_layers', type=int, default=1, choices=[1, 2, 3], help="Number of encoder layers")
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['concat', 'weighted', 'attention', 'rna_only', 'aux_only'], help="Multi-omics fusion")
    parser.add_argument('--alpha', type=float, default=0.5, help="Modality weighting factor for weighted fusion")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension")
    parser.add_argument('--out_dim', type=int, default=64, help="Embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")

    # Clustering
    parser.add_argument('--clustering_method', type=str, default='kmeans', choices=['kmeans', 'gmm', 'spectral', 'leiden'], help="Clustering algorithm")
    parser.add_argument('--cluster_restarts', type=int, default=10, help="Number of clustering restarts")

    # Output & Devices
    parser.add_argument('--device', type=str, default=None, help="'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument('--visualize', action='store_true', default=True, help="Generate plots")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory for results")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory for datasets")
    parser.add_argument('--api_url', type=str, default=None, help="Dashboard API endpoint")
    parser.add_argument('--eval_every_epoch', action='store_true', default=False, help="Evaluate KMeans on every epoch")

    # Mode-specific arguments
    parser.add_argument('--ofat_param', type=str, default='beta', choices=['beta', 'gamma', 'delta', 'lr', 'spatial_k', 'all'])
    parser.add_argument('--hpo_trials', type=int, default=30, help="Number of Optuna trials")
    parser.add_argument('--hpo_search_mode', type=str, default='standard', choices=['loss_weights', 'training', 'graph', 'architecture', 'full', 'standard'])
    parser.add_argument('--study_name', type=str, default='muse_gcn_universal_hpo', help="Name of the Optuna study")
    parser.add_argument('--ablation_type', type=str, default='fusion', choices=['backbone', 'depth', 'fusion', 'all'])

    args = parser.parse_args()

    if args.mode == 'ofat':
        from experiments.sensitivity_ofat import run_ofat_sweep
        target_dataset = args.datasets[0] if args.datasets != ['all'] else '10x_human_lymph_node_A1'
        target_seeds = [int(s) for s in args.seeds] if args.seeds else [42, 1234, 2024]
        if args.ofat_param == 'all':
            for p in ['beta', 'gamma', 'delta', 'lr', 'spatial_k']:
                run_ofat_sweep(p, dataset_name=target_dataset, seeds=target_seeds, output_dir=os.path.join(args.output_dir, "sensitivity"))
        else:
            run_ofat_sweep(args.ofat_param, dataset_name=target_dataset, seeds=target_seeds, output_dir=os.path.join(args.output_dir, "sensitivity"))

    elif args.mode == 'hpo':
        from optimization.tune_optuna import run_optuna_study
        target_datasets = args.datasets if len(args.datasets) > 1 or args.datasets[0] != 'all' else 'all'
        target_seeds = [int(s) for s in args.seeds] if args.seeds else [42, 1234, 2024]
        run_optuna_study(
            datasets=target_datasets,
            seeds=target_seeds,
            n_trials=args.hpo_trials,
            search_mode=args.hpo_search_mode,
            epochs_per_trial=args.epochs,
            study_name=args.study_name,
            storage_dir=os.path.join(args.output_dir, "optimization"),
            device=args.device
        )

    elif args.mode == 'ablation':
        from experiments.ablation_runner import (
            run_gnn_backbone_ablation,
            run_depth_ablation,
            run_fusion_ablation
        )
        target_dataset = args.datasets[0] if args.datasets != ['all'] else '10x_human_lymph_node_A1'
        target_seeds = [int(s) for s in args.seeds] if args.seeds else [42, 1234, 2024]
        abl_dir = os.path.join(args.output_dir, "ablations")

        if args.ablation_type in ['backbone', 'all']:
            run_gnn_backbone_ablation(target_dataset, target_seeds, abl_dir)
        if args.ablation_type in ['depth', 'all']:
            run_depth_ablation(target_dataset, target_seeds, abl_dir)
        if args.ablation_type in ['fusion', 'all']:
            run_fusion_ablation(target_dataset, target_seeds, abl_dir)

    else:
        # Standard Training Mode
        run_experiment(
            datasets=args.datasets if len(args.datasets) > 1 or args.datasets[0] != 'all' else 'all',
            seeds=args.seeds,
            n_seeds=args.n_seeds,
            rna_pca_comps=args.rna_pca_comps,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            spatial_k=args.spatial_k,
            feature_k_rna=args.feature_k_rna,
            feature_k_aux=args.feature_k_aux,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            dropout=args.dropout,
            gnn_type=args.gnn_type,
            num_layers=args.num_layers,
            fusion_type=args.fusion_type,
            alpha=args.alpha,
            clustering_method=args.clustering_method,
            cluster_restarts=args.cluster_restarts,
            device=args.device,
            visualize=args.visualize,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            api_url=args.api_url,
            eval_every_epoch=args.eval_every_epoch
        )

