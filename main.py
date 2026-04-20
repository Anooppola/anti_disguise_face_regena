"""
main.py - Project entry point
Routes to train, evaluate, or serve the API based on CLI arguments.
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

COMMANDS = {
    "train":    "Start GAN training with MLflow tracking",
    "evaluate": "Evaluate saved model on validation set",
    "serve":    "Start FastAPI inference server",
    "mlflow":   "Launch MLflow UI",
    "frontend": "Launch Streamlit frontend",
    "test-exp": "Log a test experiment to MLflow (for demo)",
}


def cmd_train(rest):
    from src.train import parse_args, train
    cfg = parse_args()
    train(cfg)


def cmd_evaluate(rest):
    import argparse, torch
    from pathlib import Path
    from src.model import Generator, Discriminator
    from src.data_loader import get_dataloaders
    from src.evaluate import evaluate

    p = argparse.ArgumentParser()
    p.add_argument("--masked_dir",   default="data/masked")
    p.add_argument("--unmasked_dir", default="data/unmasked")
    p.add_argument("--model",        default="saved_models/generator_best.pth")
    p.add_argument("--batch_size",   type=int, default=4)
    args = p.parse_args(rest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    D = Discriminator().to(device)

    import torch
    G.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    _, val_loader = get_dataloaders(args.masked_dir, args.unmasked_dir,
                                    batch_size=args.batch_size)
    metrics = evaluate(G, D, val_loader, device)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")


def cmd_serve(rest):
    import uvicorn, os
    uvicorn.run(
        "api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=False,
    )


def cmd_mlflow(rest):
    from mlflow_utils.run_mlflow_ui import run_mlflow_ui
    run_mlflow_ui()


def cmd_frontend(rest):
    import subprocess, sys
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "frontend/streamlit_app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
    ])


def cmd_test_exp(rest):
    from mlflow_utils.track_experiments import run_test_experiment
    run_test_experiment()


DISPATCH = {
    "train":    cmd_train,
    "evaluate": cmd_evaluate,
    "serve":    cmd_serve,
    "mlflow":   cmd_mlflow,
    "frontend": cmd_frontend,
    "test-exp": cmd_test_exp,
}


def main():
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="Anti-Disguise Face Reconstruction — entry point",
    )
    p.add_argument(
        "command",
        choices=list(COMMANDS.keys()),
        help=" | ".join(f"{k}: {v}" for k, v in COMMANDS.items()),
    )
    args, rest = p.parse_known_args()
    
    logger.info("Running command: %s", args.command)
    DISPATCH[args.command](rest)


if __name__ == "__main__":
    main()
