# ==============================================================================
# optimization/tune_optuna.py - Optuna Study Runner & Automated HPO
# ==============================================================================

import os
import json
import argparse
import pandas as pd
from typing import Optional

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner
except ImportError:
    optuna = None

from optimization.objective import MuseGCNOptimizationObjective


def run_optuna_study(
    dataset_name: str = "10x_human_lymph_node_A1",
    n_trials: int = 50,
    search_mode: str = "standard",
    metric_target: str = "silhouette",
    epochs_per_trial: int = 150,
    sampler_type: str = "tpe",
    seed: int = 42,
    study_name: str = "muse_gcn_hpo",
    storage_dir: str = "results/optimization",
    device: Optional[str] = None
):
    """
    Run automated Bayesian Optimization (TPE) study with Optuna:
      1. Proposes parameter configurations
      2. Trains and prunes underperforming trials
      3. Saves trial history to CSV and best config to JSON
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Please run: pip install optuna")

    os.makedirs(storage_dir, exist_ok=True)
    db_path = os.path.join(storage_dir, f"{study_name}.db")
    storage_url = f"sqlite:///{os.path.abspath(db_path)}"

    if sampler_type.lower() == "random":
        sampler = RandomSampler(seed=seed)
    else:
        sampler = TPESampler(seed=seed, multivariate=True)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=25)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    objective = MuseGCNOptimizationObjective(
        dataset_name=dataset_name,
        epochs=epochs_per_trial,
        seed=seed,
        metric_target=metric_target,
        search_mode=search_mode,
        device=device
    )

    print("\n" + "=" * 75)
    print(f" 🔍 LAUNCHING OPTUNA STUDY: {study_name} ".center(75, "="))
    print("=" * 75)
    print(f"• Dataset         : {dataset_name}")
    print(f"• Target Metric   : {metric_target.upper()}")
    print(f"• Trials          : {n_trials} | Sampler: {sampler_type.upper()}")
    print(f"• Search Mode     : {search_mode}")
    print(f"• Epochs / Trial  : {epochs_per_trial}")
    print(f"• Database Storage: {storage_url}")
    print("=" * 75 + "\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n" + "=" * 75)
    print(" 🏆 OPTUNA STUDY COMPLETE - BEST CONFIGURATION ".center(75, "="))
    print("=" * 75)
    print(f"Best Trial Number : #{study.best_trial.number}")
    print(f"Best Value ({metric_target}): {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  • {k:18s}: {v}")
    print("=" * 75 + "\n")

    # Save Study Results to CSV and Best Params to JSON
    df_trials = study.trials_dataframe()
    csv_path = os.path.join(storage_dir, f"{study_name}_trials.csv")
    df_trials.to_csv(csv_path, index=False)
    print(f"💾 Full trials dataframe saved to: {csv_path}")

    best_config_path = os.path.join(storage_dir, f"{study_name}_best_config.json")
    with open(best_config_path, "w") as f:
        json.dump({
            "study_name": study_name,
            "dataset": dataset_name,
            "best_value": study.best_value,
            "metric_target": metric_target,
            "best_params": study.best_params
        }, f, indent=2)
    print(f"💾 Best configuration saved to: {best_config_path}")

    # Compute Parameter Importances
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\n📊 Hyperparameter Importances:")
        for param, imp in importances.items():
            print(f"  • {param:18s}: {imp:.4f} ({imp*100:.1f}%)")
    except Exception as e:
        print(f"ℹ️ Could not compute parameter importance: {e}")

    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optuna HPO for Muse-GCN")
    parser.add_argument('--dataset', type=str, default='10x_human_lymph_node_A1')
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--search_mode', type=str, default='standard', choices=['loss_weights', 'training', 'graph', 'architecture', 'full', 'standard'])
    parser.add_argument('--metric_target', type=str, default='silhouette', choices=['silhouette', 'ari'])
    parser.add_argument('--epochs_per_trial', type=int, default=150)
    parser.add_argument('--sampler', type=str, default='tpe', choices=['tpe', 'random'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--study_name', type=str, default='muse_gcn_hpo')
    parser.add_argument('--storage_dir', type=str, default='results/optimization')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    run_optuna_study(
        dataset_name=args.dataset,
        n_trials=args.n_trials,
        search_mode=args.search_mode,
        metric_target=args.metric_target,
        epochs_per_trial=args.epochs_per_trial,
        sampler_type=args.sampler,
        seed=args.seed,
        study_name=args.study_name,
        storage_dir=args.storage_dir,
        device=args.device
    )

