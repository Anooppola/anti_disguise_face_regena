"""
track_experiments.py - Standalone script to log a test experiment to MLflow
Run this to verify your MLflow setup is working correctly.
"""

import logging
import math

import mlflow
from mlflow_utils.mlflow_utils import setup_mlflow, log_params, log_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test_experiment():
    setup_mlflow("anti-disguise-test-experiment")

    params = {
        "learning_rate": 2e-4,
        "batch_size":    4,
        "epochs":        5,
        "lambda_l1":     100,
        "lambda_percep": 10,
    }

    with mlflow.start_run(run_name="test-run"):
        log_params(params)

        for epoch in range(1, params["epochs"] + 1):
            # Simulate decreasing losses and improving quality metrics
            g_loss  = 5.0   * math.exp(-0.3 * epoch)
            d_loss  = 1.5   * math.exp(-0.2 * epoch)
            psnr    = 15.0  + 3.0  * math.log1p(epoch)
            ssim    = 0.60  + 0.05 * math.log1p(epoch)

            log_metrics(
                {"g_loss": g_loss, "d_loss": d_loss, "psnr": psnr, "ssim": ssim},
                step=epoch,
            )
            logger.info("Epoch %d | G=%.4f D=%.4f PSNR=%.2f SSIM=%.4f",
                        epoch, g_loss, d_loss, psnr, ssim)

        mlflow.set_tag("status", "test-complete")
        logger.info("Test experiment logged successfully.")


if __name__ == "__main__":
    run_test_experiment()
    print("\nOpen MLflow UI at http://127.0.0.1:5000 to view results.")
