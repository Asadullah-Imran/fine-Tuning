# ==============================================================================
# utils/data_loader.py - Dataset Registry, Download, Preprocessing
# ==============================================================================

import os
import random
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from typing import Tuple, Dict, Any, List

import scanpy as sc
import anndata
from sklearn.decomposition import PCA
import torch

DATASET_REGISTRY = {
    "10x_human_lymph_node_A1": {
        "index": 0,
        "url": "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC",
        "type": "10x",
        "gt_col": "manual-anno",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv"
    },
    "10x_human_lymph_node_D1": {
        "index": 1,
        "url": "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz",
        "type": "10x",
        "gt_col": "manual-anno",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv"
    },
    "Mouse_Brain_E11_S1": {
        "index": 2,
        "url": "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E13_S1": {
        "index": 3,
        "url": "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E15_S1": {
        "index": 4,
        "url": "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E18_S1": {
        "index": 5,
        "url": "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
}

DATASET_LIST = list(DATASET_REGISTRY.keys())

DEFAULT_SEEDS = [
    13, 2560, 641, 1892, 1173, 69, 2024, 231, 1971, 2497,
    338, 3127, 2001, 2022, 574, 2428, 999, 1187, 42, 3999
]


def set_seed(seed: int = 2024):
    """Set random seed across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def clr_normalize_each_cell(adata, inplace=True):
    """Centered Log-Ratio (CLR) normalization on protein (ADT) data."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def pca(adata, use_reps=None, n_comps=60):
    """PCA dimensionality reduction."""
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        feat_pca = pca_model.fit_transform(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
    return feat_pca


def Protein(adata):
    """Normalize ADT (Protein) data using CLR and scale it."""
    adata = clr_normalize_each_cell(adata)
    sc.pp.scale(adata)
    return adata


def tfidf(X):
    """Compute TF-IDF matrix in CSR sparse format."""
    idf = X.shape[0] / X.sum(axis=0)
    if sp.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return sp.csr_matrix(tf.multiply(idf))
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def preprocess_with_rna_pca(adata_RNA, adata_omics2, dataset_name, rna_pca_comps=60):
    """
    Preprocessing with RNA PCA dimensionality reduction:
      - RNA: Filtered, HVG, normalized, log1p, scaled, then PCA reduced.
      - ADT/ATAC: CLR / TF-IDF + PCA.
    """
    sc.pp.filter_genes(adata_RNA, min_cells=10)
    sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.scale(adata_RNA)

    adata_RNA_high = adata_RNA[:, adata_RNA.var['highly_variable']].copy()
    actual_pca_comps = min(rna_pca_comps, adata_RNA_high.n_vars, adata_RNA_high.n_obs - 1)
    RNA_expression = pca(adata_RNA_high, n_comps=actual_pca_comps)

    if dataset_name.startswith("10x"):
        adata_omics2 = adata_omics2[adata_RNA.obs_names].copy()
        adata_omics2 = Protein(adata_omics2)
        ADT_expression = adata_omics2.X
    else:
        adata_omics2 = adata_omics2[adata_RNA.obs_names].copy()
        adata_omics2.X = tfidf(adata_omics2.X)
        sc.pp.normalize_per_cell(adata_omics2, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_omics2)
        n_comps = min(60, adata_omics2.shape[1])
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_comps)
        ADT_expression = adata_omics2.obsm['feat']

    if scipy.sparse.issparse(ADT_expression):
        ADT_expression = ADT_expression.toarray()

    return RNA_expression, ADT_expression


def load_dataset(dataset_name: str, data_dir: str = "data", rna_pca_comps: int = 60) -> Dict[str, Any]:
    """
    Load dataset files (downloading if missing), preprocess, and return clean dictionary.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not recognized in registry.")

    cfg = DATASET_REGISTRY[dataset_name]
    base = os.path.join(data_dir, dataset_name)
    os.makedirs(base, exist_ok=True)

    rna_path = os.path.join(base, "adata_RNA.h5ad")
    other_path = os.path.join(base, cfg["other_file"])
    annotation_path = os.path.join(base, cfg["anno_file"])

    if not os.path.exists(rna_path) or not os.path.exists(other_path) or not os.path.exists(annotation_path):
        print(f"[DataLoader] Downloading dataset '{dataset_name}' into: {base}")
        gdown_cmd = ".venv/bin/gdown" if os.path.exists(".venv/bin/gdown") else "gdown"
        os.system(f'{gdown_cmd} --folder "{cfg["url"]}" --output "{base}"')

    adata_RNA = sc.read_h5ad(rna_path)
    adata_omics2 = sc.read_h5ad(other_path)
    adata_RNA.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    anno_df = pd.read_csv(annotation_path, index_col=0)
    true_labels = anno_df[cfg["gt_col"]].values
    adata_RNA.obs['ground_truth'] = pd.Categorical(true_labels)
    num_clusters = len(np.unique(true_labels))

    RNA_expr, ADT_expr = preprocess_with_rna_pca(
        adata_RNA, adata_omics2, dataset_name, rna_pca_comps=rna_pca_comps
    )
    cell_positions = adata_RNA.obsm['spatial']

    return {
        'dataset_name': dataset_name,
        'adata_RNA': adata_RNA,
        'adata_omics2': adata_omics2,
        'true_labels': true_labels,
        'num_clusters': num_clusters,
        'RNA_expr': RNA_expr,
        'ADT_expr': ADT_expr,
        'cell_positions': cell_positions,
        'n_spots': adata_RNA.n_obs
    }

