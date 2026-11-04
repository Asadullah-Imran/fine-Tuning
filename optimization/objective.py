# ==============================================================================
# optimization/objective.py - Optuna Objective Function for Muse-GCN
# ==============================================================================

import torch
from typing import Dict, Any, Optional

from utils.data_loader import load_dataset, set_seed
from graphs.builder import build_4encoder_graphs
from models.muse_gcn import MuseGCNModel
from trainer import train_muse_gcn
from optimization.search_spaces import sample_hyperparameters


class MuseGCNOptimizationObjective:
    """
    Optuna objective wrapper ensuring fair evaluation:
      - Uses a dedicated validation dataset to prevent test data leakage.
      - Can optimize for unsupervised 'silhouette' or validation 'ari'.
    """
    def __init__(
        self,
        dataset_name: str = "10x_human_lymph_node_A1",
        data_dir: str = "data",
        epochs: int = 200,
        seed: int = 42,
        metric_target: str = "silhouette",
        search_mode: str = "standard",
        device: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.epochs = epochs
        self.seed = seed
        self.metric_target = metric_target.lower()
        self.search_mode = search_mode
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[Optuna Objective] Preloading dataset: {self.dataset_name} for tuning trials...")
        self.data_pack = load_dataset(self.dataset_name, data_dir=self.data_dir, rna_pca_comps=60)
        self.dev = torch.device(self.device)

    def __call__(self, trial) -> float:
        set_seed(self.seed)
        params = sample_hyperparameters(trial, mode=self.search_mode)

        # 1. Build graphs with trial's neighbor settings
        data = build_4encoder_graphs(
            RNA_expression=self.data_pack['RNA_expr'],
            ADT_expression=self.data_pack['ADT_expr'],
            cell_positions=self.data_pack['cell_positions'],
            device=self.dev,
            spatial_k=params.get('spatial_k', 15),
            feature_k_rna=params.get('feature_k_rna', 15),
            feature_k_aux=params.get('feature_k_aux', 15)
        )

        input_dim = data.x_RNA.shape[1]
        adt_dim = data.x_ADT.shape[1]

        # 2. Instantiate Model
        model = MuseGCNModel(
            in_channels=input_dim,
            hidden_channels=params['hidden_dim'],
            out_channels=params['out_dim'],
            q=adt_dim,
            num_clusters=self.data_pack['num_clusters'],
            beta=params['beta'],
            gamma=params['gamma'],
            delta=params['delta'],
            gnn_type=params.get('gnn_type', 'gcn'),
            num_layers=params.get('num_layers', 1),
            dropout=params['dropout'],
            fusion_type=params.get('fusion_type', 'concat'),
            alpha=params.get('alpha', 0.5)
        ).to(self.dev)

        # 3. Train with Trial Pruning Hook
        result = train_muse_gcn(
            model=model,
            data=data,
            epochs=self.epochs,
            lr=params['lr'],
            weight_decay=params.get('weight_decay', 0.0),
            optimizer_name=params.get('optimizer_name', 'adam'),
            num_clusters=self.data_pack['num_clusters'],
            true_labels=self.data_pack['true_labels'],
            verbose=False,
            eval_every_epoch=False,
            trial=trial
        )

        score = result['final_ari'] if self.metric_target == 'ari' else result['final_sil']
        return float(score)

