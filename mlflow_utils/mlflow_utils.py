"""
mlflow_utils.py - MLflow helper utilities
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
    """
    Configure MLflow tracking URI and set (or create) the experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s | Experiment: %s", tracking_uri, experiment_name)


def log_params(params: Dict[str, Any]):
    """Log a flat dict of hyper-parameters to the active MLflow run."""
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: int = None):
    """Log a dict of metrics to the active MLflow run."""
    mlflow.log_metrics(metrics, step=step)


def save_model_artifact(model: nn.Module, save_dir: str, name: str):
    """
    Save model state_dict to disk and log as an MLflow artifact.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.pth"
    torch.save(model.state_dict(), str(path))
    mlflow.log_artifact(str(path), artifact_path="models")
    logger.info("Model artifact saved and logged: %s", path)


def load_model_from_run(run_id: str, artifact_path: str, model: nn.Module, device: torch.device):
    """
    Load a model state_dict from a specific MLflow run artifact.
    """
    artifact_uri = f"runs:/{run_id}/{artifact_path}"
    local_path   = mlflow.artifacts.download_artifacts(artifact_uri)
    state        = torch.load(local_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded model from MLflow run %s / %s", run_id, artifact_path)
    return model
