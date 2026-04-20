"""
train.py - Full training loop for Pix2Pix GAN with MLflow tracking
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

import mlflow
import mlflow.pytorch

from src.model import Generator, Discriminator, VGGPerceptualLoss
from src.data_loader import get_dataloaders
from src.evaluate import evaluate
from mlflow_utils.mlflow_utils import setup_mlflow, log_metrics, save_model_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Training configuration ────────────────────────────────────────────────────

DEFAULTS = {
    "masked_dir":    "data/masked",
    "unmasked_dir":  "data/unmasked",
    "image_size":    256,
    "batch_size":    4,
    "epochs":        50,
    "lr":            2e-4,
    "beta1":         0.5,
    "beta2":         0.999,
    "lambda_l1":     100.0,
    "lambda_percep": 10.0,
    "val_split":     0.1,
    "save_dir":      "saved_models",
    "save_every":    10,
    "experiment":    "anti-disguise-pix2pix",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train Pix2Pix GAN")
    for key, default in DEFAULTS.items():
        p.add_argument(f"--{key}", type=type(default), default=default)
    return p.parse_args()


# ─── Loss functions ────────────────────────────────────────────────────────────

def adversarial_loss(pred: torch.Tensor, is_real: bool) -> torch.Tensor:
    """LSGAN adversarial loss (MSE rather than BCE for stability)."""
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return nn.MSELoss()(pred, target)


# ─── Training step ─────────────────────────────────────────────────────────────

def train_epoch(
    G, D, vgg_loss,
    opt_G, opt_D,
    loader,
    device,
    lambda_l1, lambda_percep,
    epoch: int,
):
    G.train(); D.train()
    g_total = d_total = 0.0
    l1_criterion = nn.L1Loss()

    for step, (masked, unmasked) in enumerate(loader):
        masked   = masked.to(device)
        unmasked = unmasked.to(device)

        # ── Discriminator update ──
        opt_D.zero_grad()
        fake = G(masked).detach()

        real_pred = D(masked, unmasked)
        fake_pred = D(masked, fake)
        d_loss = 0.5 * (adversarial_loss(real_pred, True) +
                         adversarial_loss(fake_pred, False))
        d_loss.backward()
        opt_D.step()

        # ── Generator update ──
        opt_G.zero_grad()
        fake = G(masked)
        fake_pred = D(masked, fake)

        loss_adv    = adversarial_loss(fake_pred, True)
        loss_l1     = l1_criterion(fake, unmasked) * lambda_l1
        loss_percep = vgg_loss(fake, unmasked)     * lambda_percep
        g_loss = loss_adv + loss_l1 + loss_percep
        g_loss.backward()
        opt_G.step()

        g_total += g_loss.item()
        d_total += d_loss.item()

        if step % 20 == 0:
            logger.info(
                "Epoch %d | Step %d/%d | G_loss=%.4f | D_loss=%.4f",
                epoch, step, len(loader), g_loss.item(), d_loss.item()
            )

    n = len(loader)
    return g_total / n, d_total / n


# ─── Main training function ────────────────────────────────────────────────────

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(
        masked_dir   = cfg.masked_dir,
        unmasked_dir = cfg.unmasked_dir,
        image_size   = cfg.image_size,
        batch_size   = cfg.batch_size,
        val_split    = cfg.val_split,
        augment_train= True,
    )

    # Models
    G = Generator().to(device)
    D = Discriminator().to(device)
    vgg = VGGPerceptualLoss(device)

    # Optimizers
    opt_G = Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_D = Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    best_psnr = 0.0
    best_model_path = os.path.join(cfg.save_dir, "generator_best.pth")

    # ── MLflow run ──
    setup_mlflow(cfg.experiment)
    run_params = {k: v for k, v in vars(cfg).items()
                  if k not in ("masked_dir", "unmasked_dir")}

    with mlflow.start_run():
        mlflow.log_params(run_params)
        mlflow.set_tag("device", str(device))

        for epoch in range(1, cfg.epochs + 1):
            g_loss, d_loss = train_epoch(
                G, D, vgg, opt_G, opt_D,
                train_loader, device,
                cfg.lambda_l1, cfg.lambda_percep, epoch,
            )

            metrics = evaluate(G, D, val_loader, device)
            metrics["g_loss"] = g_loss
            metrics["d_loss"] = d_loss

            log_metrics(metrics, step=epoch)
            logger.info(
                "Epoch %d → G=%.4f D=%.4f PSNR=%.2f SSIM=%.4f",
                epoch, g_loss, d_loss, metrics["psnr"], metrics["ssim"]
            )

            # Checkpoint every N epochs
            if epoch % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.save_dir, f"generator_epoch{epoch:03d}.pth")
                torch.save(G.state_dict(), ckpt_path)
                logger.info("Checkpoint saved: %s", ckpt_path)

            # Save best model
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                torch.save(G.state_dict(), best_model_path)
                mlflow.log_metric("best_psnr", best_psnr, step=epoch)
                logger.info("New best model (PSNR=%.2f): %s", best_psnr, best_model_path)

        # Log final model artifact
        save_model_artifact(G, cfg.save_dir, "generator_final")
        logger.info("Training complete. Best PSNR: %.2f dB", best_psnr)


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
