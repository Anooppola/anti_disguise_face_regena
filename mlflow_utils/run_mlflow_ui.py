"""
run_mlflow_ui.py - Start the MLflow tracking server
"""

import os
import subprocess
import sys


def run_mlflow_ui(
    host: str = "0.0.0.0",
    port: int = 5000,
    backend_store: str = "sqlite:///mlflow.db",
    artifact_root: str = "./mlruns",
):
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store,
        "--default-artifact-root", artifact_root,
    ]
    print(f"Starting MLflow UI at http://{host}:{port}")
    print("Press CTRL+C to stop.\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_mlflow_ui()
