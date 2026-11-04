# MUSE-GCN Systematic Fine-Tuning & Experimental Suite

This workspace contains the complete modular implementation and optimization framework for **MUSE-GCN** (*Multi-omics Unified Spatial Embedding Graph Convolutional Network*), developed for FYDP research in spatial multi-omics domain identification.

---

## 📁 Repository Structure

```
fine-Tuning/
├── train.py                  # Unified execution entrypoint (train, ofat, hpo, ablation)
├── trainer.py                # Core training loop with Silhouette/ARI logging & pruning
├── requirements.txt          # Python dependencies
├── configs/                  # YAML experiment presets
│   ├── baseline.yaml
│   ├── optuna_hpo.yaml
│   └── ablations.yaml
├── models/                   # Modular Neural Architecture
│   ├── backbones.py          # GCN, GAT, GraphSAGE, GIN with 1-3 layers & residual links
│   ├── fusion.py             # Concat, Weighted (α), Attention, and Ablation fusion
│   └── muse_gcn.py           # MuseGCN model with configurable loss weights (β, γ, δ)
├── graphs/
│   └── builder.py            # Spatial kNN & feature similarity graph builder
├── clustering/
│   └── cluster_engine.py     # KMeans, GMM, Leiden, Louvain, Spectral clustering
├── optimization/
│   ├── search_spaces.py      # Optuna trial parameter distributions
│   ├── objective.py          # Optuna objective with cross-dataset isolation
│   └── tune_optuna.py        # Bayesian optimization (TPE) study runner
├── experiments/
│   ├── sensitivity_ofat.py   # One-Factor-At-A-Time sensitivity runner & curves
│   └── ablation_runner.py    # GNN backbone, layer depth, and fusion ablation runner
└── utils/
    ├── data_loader.py        # Dataset registry, download, and preprocessing
    ├── metrics.py            # ARI, NMI, Silhouette, AMI, CHI, DBI calculation
    ├── visualization.py      # Curves, spatial domain maps, UMAPs, violins
    └── dashboard_exporter.py # Standardized JSON export & live API pusher
```

---

## 🚀 Quickstart Commands

### 1. Reproduce Frozen Baseline
```bash
python3 train.py --datasets all --seeds 42 1234 2024 --epochs 350 --output_dir results/baseline
```

### 2. Run Hyperparameter Sensitivity Sweeps (OFAT)
```bash
# Sweep beta: [5, 10, 15, 25, 50, 100]
python3 train.py --mode ofat --ofat_param beta --datasets 10x_human_lymph_node_A1

# Sweep gamma: [0.1, 1, 5, 10, 20, 30]
python3 train.py --mode ofat --ofat_param gamma --datasets 10x_human_lymph_node_A1

# Run all sweeps (beta, gamma, delta, lr, spatial_k)
python3 train.py --mode ofat --ofat_param all --datasets 10x_human_lymph_node_A1
```

### 3. Automated Bayesian Optimization (Optuna TPE)
```bash
# Optimize loss weights and training hyperparameters
python3 train.py --mode hpo --hpo_search_mode standard --hpo_trials 30 --datasets 10x_human_lymph_node_A1
```

### 4. Architectural & Modality Ablations
```bash
# Modality Fusion: RNA-only vs Aux-only vs Concat vs Weighted (α) vs Attention
python3 train.py --mode ablation --ablation_type fusion --datasets 10x_human_lymph_node_A1

# Backbone Comparison: GCN vs GAT vs GraphSAGE
python3 train.py --mode ablation --ablation_type backbone --datasets 10x_human_lymph_node_A1

# Depth Comparison: 1 layer vs 2 layers vs 3 layers
python3 train.py --mode ablation --ablation_type depth --datasets 10x_human_lymph_node_A1
```

### 5. Final Multi-Seed Benchmark Evaluation
```bash
python3 train.py --datasets all --seeds 13 42 641 1234 1892 2024 2560 3127 --epochs 350 --output_dir results/final_benchmark
```

