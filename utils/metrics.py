# ==============================================================================
# utils/metrics.py - Evaluation Metrics for Spatial Clustering
# ==============================================================================

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from typing import Dict, Any, Optional


def compute_all_metrics(
    embeddings: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive clustering and embedding quality metrics:
      - Supervised: ARI, NMI, AMI, Homogeneity, V-measure
      - Unsupervised: Silhouette, Calinski-Harabasz (CHI), Davies-Bouldin (DBI)
    """
    unique_preds = len(np.unique(pred_labels))
    if unique_preds <= 1:
        sil = -1.0
        chi = 0.0
        dbi = 999.0
    else:
        try:
            sil = float(silhouette_score(embeddings, pred_labels))
        except Exception:
            sil = 0.0
        try:
            chi = float(calinski_harabasz_score(embeddings, pred_labels))
        except Exception:
            chi = 0.0
        try:
            dbi = float(davies_bouldin_score(embeddings, pred_labels))
        except Exception:
            dbi = 0.0

    metrics = {
        'Silhouette': sil,
        'CHI': chi,
        'DBI': dbi
    }

    if true_labels is not None:
        metrics['ARI'] = float(adjusted_rand_score(true_labels, pred_labels))
        metrics['NMI'] = float(normalized_mutual_info_score(true_labels, pred_labels))
        metrics['AMI'] = float(adjusted_mutual_info_score(true_labels, pred_labels))
        metrics['Homogeneity'] = float(homogeneity_score(true_labels, pred_labels))
        metrics['V-measure'] = float(v_measure_score(true_labels, pred_labels))
    else:
        metrics['ARI'] = 0.0
        metrics['NMI'] = 0.0
        metrics['AMI'] = 0.0
        metrics['Homogeneity'] = 0.0
        metrics['V-measure'] = 0.0

    return metrics

