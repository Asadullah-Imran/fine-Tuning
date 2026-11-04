# ==============================================================================
# optimization/search_spaces.py - Optuna Hyperparameter & Architecture Spaces
# ==============================================================================

from typing import Dict, Any


def sample_hyperparameters(trial, mode: str = "standard") -> Dict[str, Any]:
    """
    Sample hyperparameters based on search recommendations in tune.md:
      - 'loss_weights': Focuses on beta, gamma, delta, lr
      - 'training': Focuses on lr, weight_decay, hidden_dim, out_dim, dropout
      - 'graph': Focuses on spatial_k, feature_k_rna, feature_k_aux
      - 'architecture': GNN backbones, num_layers, fusion_type, alpha
      - 'full': Joint search over all tunable dimensions
    """
    mode = mode.lower()
    params = {}

    # Loss Function Weights (High Priority in tune.md)
    if mode in ['loss_weights', 'standard', 'full']:
        params['beta'] = trial.suggest_float('beta', 5.0, 100.0, log=True)
        params['gamma'] = trial.suggest_float('gamma', 0.1, 30.0, log=True)
        params['delta'] = trial.suggest_float('delta', 0.1, 10.0, log=True)
    else:
        params['beta'] = 25.0
        params['gamma'] = 10.0
        params['delta'] = 1.0

    # Training Hyperparameters
    if mode in ['training', 'standard', 'full']:
        params['lr'] = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        params['out_dim'] = trial.suggest_categorical('out_dim', [32, 64, 128])
        params['optimizer_name'] = trial.suggest_categorical('optimizer_name', ['adam', 'adamw'])
    else:
        params['lr'] = 1e-3
        params['weight_decay'] = 0.0
        params['dropout'] = 0.0
        params['hidden_dim'] = 512
        params['out_dim'] = 64
        params['optimizer_name'] = 'adam'

    # Graph Parameters
    if mode in ['graph', 'full']:
        params['spatial_k'] = trial.suggest_categorical('spatial_k', [4, 6, 8, 10, 12, 15, 20])
        params['feature_k_rna'] = trial.suggest_categorical('feature_k_rna', [4, 6, 8, 10, 12, 15, 20])
        params['feature_k_aux'] = trial.suggest_categorical('feature_k_aux', [4, 6, 8, 10, 12, 15, 20])
    else:
        params['spatial_k'] = 15
        params['feature_k_rna'] = 15
        params['feature_k_aux'] = 15

    # Architecture & Fusion
    if mode in ['architecture', 'full']:
        params['gnn_type'] = trial.suggest_categorical('gnn_type', ['gcn', 'gat', 'sage'])
        params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        params['fusion_type'] = trial.suggest_categorical('fusion_type', ['concat', 'weighted', 'attention'])
        if params['fusion_type'] == 'weighted':
            params['alpha'] = trial.suggest_float('alpha', 0.1, 0.9, step=0.1)
        else:
            params['alpha'] = 0.5
    else:
        params['gnn_type'] = 'gcn'
        params['num_layers'] = 1
        params['fusion_type'] = 'concat'
        params['alpha'] = 0.5

    return params

