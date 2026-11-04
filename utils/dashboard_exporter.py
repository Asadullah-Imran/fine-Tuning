# ==============================================================================
# utils/dashboard_exporter.py - Standardized JSON Exporter & Live Dashboard Pusher
# ==============================================================================

import os
import json
import numpy as np
from typing import Dict, Any, Optional


def make_json_serializable(obj):
    """Recursively converts NumPy types, tensors, and arrays into native JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj


def export_dashboard_experiment(
    model_id: str,
    model_name: str,
    dataset_name: str,
    seed: int,
    metrics_dict: Dict[str, Any],
    training_results: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    embeddings_data: Optional[Dict[str, Any]] = None,
    output_dir: str = "results",
    api_url: Optional[str] = None
):
    """
    Export standardized JSON artifacts & push directly to Dashboard via API.
    """
    exp_dir = os.path.join(output_dir, "experiments", model_id, dataset_name, f"seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    loss_hist = training_results.get("loss_history", [])
    recon_hist = training_results.get("recon_loss_history", [])
    spatial_hist = training_results.get("spatial_loss_history", [])
    reg_hist = training_results.get("reg_loss_history", [])
    sil_hist = training_results.get("epoch_sil_history", [])
    ari_hist = training_results.get("epoch_ari_history", [])

    history_points = []
    for ep in range(len(loss_hist)):
        history_points.append({
            "epoch": ep + 1,
            "losses": {
                "total_loss": float(loss_hist[ep]) if ep < len(loss_hist) else 0.0,
                "reconstruction_loss": float(recon_hist[ep]) if ep < len(recon_hist) else 0.0,
                "spatial_loss": float(spatial_hist[ep]) if ep < len(spatial_hist) else 0.0,
                "reg_loss": float(reg_hist[ep]) if ep < len(reg_hist) else 0.0,
            },
            "metrics": {
                "Silhouette": float(sil_hist[ep]) if ep < len(sil_hist) else 0.0,
                "ARI": float(ari_hist[ep]) if ep < len(ari_hist) else 0.0,
            },
            "total_loss": float(loss_hist[ep]) if ep < len(loss_hist) else None,
            "silhouette": float(sil_hist[ep]) if ep < len(sil_hist) else None,
            "ari": float(ari_hist[ep]) if ep < len(ari_hist) else None,
        })

    safe_metrics = make_json_serializable(metrics_dict)
    safe_hyperparameters = make_json_serializable(hyperparameters)
    safe_history = make_json_serializable(history_points)
    safe_embeddings = make_json_serializable(embeddings_data or {})

    # Save local JSON files
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "dataset": dataset_name,
            "seed": int(seed),
            "metrics": safe_metrics
        }, f, indent=2)

    with open(os.path.join(exp_dir, "curves.json"), "w") as f:
        json.dump({"history": safe_history}, f, indent=2)

    with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "dataset": dataset_name,
            "seed": int(seed),
            "hyperparameters": safe_hyperparameters
        }, f, indent=2)

    if api_url:
        try:
            import urllib.request
            payload = {
                "modelId": model_id,
                "modelName": model_name,
                "datasetId": dataset_name,
                "datasetName": dataset_name.replace("_", " ").title(),
                "seed": int(seed),
                "finalMetrics": safe_metrics,
                "history": safe_history,
                "embeddingsData": safe_embeddings,
                "modelMetadata": {
                    "architecture": "Muse-GCN Modular Framework",
                    "hyperparameters": safe_hyperparameters,
                }
            }
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    print(f"🚀 Sent results to Dashboard API ({api_url})")
        except Exception as e:
            print(f"ℹ️ Dashboard API push skipped: {e}")

