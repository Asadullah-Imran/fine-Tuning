# ==============================================================================
# graphs/builder.py - Flexible Multi-Omics Spatial & Feature Graph Builder
# ==============================================================================

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import torch
from torch_geometric.data import Data
from typing import Optional, Tuple


class Dual4GraphData(Data):
    """PyG Data container holding 4 graphs: RNA & Aux Similarity and Spatial graphs."""
    def __init__(
        self,
        x_RNA,
        x_ADT,
        sim_edge_index_rna,
        sim_edge_weight_rna,
        dist_edge_index_rna,
        dist_edge_weight_rna,
        sim_edge_index_aux,
        sim_edge_weight_aux,
        dist_edge_index_aux,
        dist_edge_weight_aux
    ):
        super().__init__()
        self.x_RNA = x_RNA
        self.x_ADT = x_ADT
        self.sim_edge_index_rna = sim_edge_index_rna
        self.sim_edge_weight_rna = sim_edge_weight_rna
        self.dist_edge_index_rna = dist_edge_index_rna
        self.dist_edge_weight_rna = dist_edge_weight_rna
        self.sim_edge_index_aux = sim_edge_index_aux
        self.sim_edge_weight_aux = sim_edge_weight_aux
        self.dist_edge_index_aux = dist_edge_index_aux
        self.dist_edge_weight_aux = dist_edge_weight_aux


def _build_feature_similarity_graph(
    features: np.ndarray,
    k_neighbors: int = 15,
    metric: str = "cosine",
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct symmetric kNN feature-similarity graph with continuous weights.
    """
    n_samples = len(features)
    k = min(k_neighbors, n_samples - 1)

    if metric == "cosine":
        sim_matrix = cosine_similarity(features)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(features)
        _, indices = nbrs.kneighbors(features)
    elif metric == "euclidean":
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(features)
        distances, indices = nbrs.kneighbors(features)
        # Convert euclidean distance to Gaussian kernel similarity
        sigma = np.median(distances) + 1e-8
        sim_matrix = np.exp(- (distances ** 2) / (2 * sigma ** 2))
    elif metric == "correlation":
        corr = np.corrcoef(features)
        sim_matrix = np.nan_to_num(corr, nan=0.0)
        # k highest correlations
        indices = np.argsort(-sim_matrix, axis=1)[:, :k + 1]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    adj = np.zeros((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        for j in indices[i][1:]:
            adj[i, j] = 1
            adj[j, i] = 1

    nonzero_idx = np.nonzero(adj)
    edge_index = torch.tensor(np.array(nonzero_idx), dtype=torch.long).to(device)

    if metric == "cosine":
        edge_weight = torch.tensor(sim_matrix[adj > 0], dtype=torch.float).to(device)
    else:
        edge_weight = torch.tensor(sim_matrix[nonzero_idx[0], nonzero_idx[1]], dtype=torch.float).to(device)

    return edge_index, edge_weight


def build_4encoder_graphs(
    RNA_expression: np.ndarray,
    ADT_expression: np.ndarray,
    cell_positions: np.ndarray,
    device: str = 'cpu',
    spatial_k: int = 15,
    feature_k_rna: Optional[int] = None,
    feature_k_aux: Optional[int] = None,
    num_neighbors: Optional[int] = None,
    feature_metric: str = 'cosine',
    include_self_loops: bool = False
) -> Dual4GraphData:
    """
    Construct 4 graphs with fine-grained parameter control:
      1. RNA Spatial Distance Graph (spatial_k)
      2. Aux Spatial Distance Graph (spatial_k)
      3. RNA Feature Similarity Graph (feature_k_rna)
      4. Aux Feature Similarity Graph (feature_k_aux)
    """
    # Backward compatibility with single num_neighbors flag
    if num_neighbors is not None:
        spatial_k = num_neighbors
        feature_k_rna = num_neighbors
        feature_k_aux = num_neighbors
    else:
        if feature_k_rna is None:
            feature_k_rna = spatial_k
        if feature_k_aux is None:
            feature_k_aux = spatial_k

    n_samples = len(cell_positions)
    sk = min(spatial_k, n_samples - 1)

    # 1. Spatial KNN Distance Graph
    knn_graph = kneighbors_graph(
        cell_positions,
        n_neighbors=sk,
        mode='distance',
        include_self=include_self_loops
    )
    knn_graph = knn_graph.maximum(knn_graph.T)

    dist_edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long).to(device)
    dist_edge_weight = torch.tensor(knn_graph.data, dtype=torch.float).to(device)

    # 2. RNA Feature Similarity Graph
    sim_edge_index_rna, sim_edge_weight_rna = _build_feature_similarity_graph(
        RNA_expression, k_neighbors=feature_k_rna, metric=feature_metric, device=device
    )

    # 3. Aux Feature Similarity Graph
    sim_edge_index_aux, sim_edge_weight_aux = _build_feature_similarity_graph(
        ADT_expression, k_neighbors=feature_k_aux, metric=feature_metric, device=device
    )

    x_RNA = torch.tensor(RNA_expression, dtype=torch.float).to(device)
    x_ADT = torch.tensor(ADT_expression, dtype=torch.float).to(device)

    return Dual4GraphData(
        x_RNA=x_RNA,
        x_ADT=x_ADT,
        sim_edge_index_rna=sim_edge_index_rna,
        sim_edge_weight_rna=sim_edge_weight_rna,
        dist_edge_index_rna=dist_edge_index,
        dist_edge_weight_rna=dist_edge_weight,
        sim_edge_index_aux=sim_edge_index_aux,
        sim_edge_weight_aux=sim_edge_weight_aux,
        dist_edge_index_aux=dist_edge_index,
        dist_edge_weight_aux=dist_edge_weight
    )

