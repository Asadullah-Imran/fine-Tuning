# ==============================================================================
# clustering/cluster_engine.py - Modular Clustering Algorithms
# ==============================================================================

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import scanpy as sc
import anndata
from typing import Optional


def cluster_embeddings(
    embeddings: np.ndarray,
    num_clusters: int,
    method: str = 'kmeans',
    random_state: int = 42,
    n_init: int = 10,
    resolution: float = 1.0
) -> np.ndarray:
    """
    Modular clustering interface supporting:
      - 'kmeans': KMeans with configurable restarts (n_init)
      - 'gmm': Gaussian Mixture Model
      - 'spectral': Spectral Clustering
      - 'leiden': Graph-based community detection via Scanpy
      - 'louvain': Modularity-based community detection via Scanpy
    """
    method = method.lower()

    if method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=random_state)
        return kmeans.fit_predict(embeddings)

    elif method == 'gmm':
        gmm = GaussianMixture(n_components=num_clusters, n_init=n_init, random_state=random_state)
        return gmm.fit_predict(embeddings)

    elif method == 'spectral':
        spectral = SpectralClustering(
            n_clusters=num_clusters,
            n_init=n_init,
            random_state=random_state,
            affinity='nearest_neighbors'
        )
        return spectral.fit_predict(embeddings)

    elif method in ['leiden', 'louvain']:
        adata_tmp = anndata.AnnData(X=embeddings)
        sc.pp.neighbors(adata_tmp, use_rep='X', n_neighbors=15)
        if method == 'leiden':
            try:
                sc.tl.leiden(adata_tmp, resolution=resolution, random_state=random_state)
                preds = adata_tmp.obs['leiden'].astype(int).values
            except Exception:
                # Fallback to KMeans if leidenalg is not installed
                kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=random_state)
                return kmeans.fit_predict(embeddings)
        else:
            try:
                sc.tl.louvain(adata_tmp, resolution=resolution, random_state=random_state)
                preds = adata_tmp.obs['louvain'].astype(int).values
            except Exception:
                kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=random_state)
                return kmeans.fit_predict(embeddings)
        return preds

    else:
        raise ValueError(f"Unsupported clustering method: {method}. Choose from 'kmeans', 'gmm', 'spectral', 'leiden', 'louvain'.")

