# ==============================================================================
# optimization/objective.py - Robust Multi-Dataset Multi-Seed Optuna Objective
# ==============================================================================

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union

from utils.data_loader import load_dataset, set_seed, DATASET_LIST
from graphs.builder import build_4encoder_graphs
from models.muse_gcn import MuseGCNModel
from trainer import train_muse_gcn
from optimization.search_spaces import sample_hyperparameters


class MuseGCNOptimizationObjective:
    """
    Robust Multi-Dataset, Multi-Seed Optuna Objective:
      - Evaluates each trial across multiple datasets and multiple seeds.
      - Maximizes the overall MEAN score across all evaluations.
      - Ensures the discovered hyperparameters are universally robust.
    """
    def __init__(
        self,
        datasets: Union[str, List[str]] = "10x_human_lymph_node_A1",
        seeds: Optional[List[int]] = None,
        data_dir: str = "data",
        epochs: int = 150,
        metric_target: str = "silhouette",
        search_mode: str = "standard",
        device: Optional[str] = None
    ):
        if datasets == "all" or datasets is None:
            self.dataset_names = DATASET_LIST
        elif isinstance(datasets, str):
            self.dataset_names = [datasets]
        else:
            self.dataset_names = list(datasets)

        self.seeds = seeds if seeds is not None and len(seeds) > 0 else [42, 1234, 2024]
        self.data_dir = data_dir
        self.epochs = epochs
        self.metric_target = metric_target.lower()
        self.search_mode = search_mode
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dev = torch.device(self.device)

        print(f"\n[Optuna Objective] Preloading {len(self.dataset_names)} dataset(s): {self.dataset_names}")
        self.loaded_data = {}
        for d_name in self.dataset_names:
            self.loaded_data[d_name] = load_dataset(d_name, data_dir=self.data_dir, rna_pca_comps=60)
        print(f"[Optuna Objective] Evaluating each trial across {len(self.seeds)} seed(s): {self.seeds}\n")

    def __call__(self, trial) -> float:
        params = sample_hyperparameters(trial, mode=self.search_mode)
        trial_scores = []

        for d_name in self.dataset_names:
            pack = self.loaded_data[d_name]

            # 1. Build graphs for current dataset with trial's neighbor settings
            data = build_4encoder_graphs(
                RNA_expression=pack['RNA_expr'],
                ADT_expression=pack['ADT_expr'],
                cell_positions=pack['cell_positions'],
                device=self.dev,
                spatial_k=params.get('spatial_k', 15),
                feature_k_rna=params.get('feature_k_rna', 15),
                feature_k_aux=params.get('feature_k_aux', 15)
            )

            input_dim = data.x_RNA.shape[1]
            adt_dim = data.x_ADT.shape[1]

            for seed in self.seeds:
                set_seed(seed)

                # 2. Instantiate Model
                model = MuseGCNModel(
                    in_channels=input_dim,
                    hidden_channels=params['hidden_dim'],
                    out_channels=params['out_dim'],
                    q=adt_dim,
                    num_clusters=pack['num_clusters'],
                    beta=params['beta'],
                    gamma=params['gamma'],
                    delta=params['delta'],
                    gnn_type=params.get('gnn_type', 'gcn'),
                    num_layers=params.get('num_layers', 1),
                    dropout=params['dropout'],
                    fusion_type=params.get('fusion_type', 'concat'),
                    alpha=params.get('alpha', 0.5)
                ).to(self.dev)

                # 3. Train
                result = train_muse_gcn(
                    model=model,
                    data=data,
                    epochs=self.epochs,
                    lr=params['lr'],
                    weight_decay=params.get('weight_decay', 0.0),
                    optimizer_name=params.get('optimizer_name', 'adam'),
                    num_clusters=pack['num_clusters'],
                    true_labels=pack['true_labels'],
                    verbose=False,
                    eval_every_epoch=False,
                    trial=None  # Multi-run evaluation: complete all seeds to get a reliable average
                )

                score = result['final_ari'] if self.metric_target == 'ari' else result['final_sil']
                trial_scores.append(score)

        mean_score = float(np.mean(trial_scores))
        std_score = float(np.std(trial_scores))
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("std_score", std_score)
        trial.set_user_attr("all_scores", trial_scores)

        return mean_score
