import argparse
import os
import sys
import time

# Fix import path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.optim import Adam  # type: ignore
from torch.optim.lr_scheduler import StepLR  # type: ignore
import torchvision.utils as vutils  # type: ignore
from torchvision.models import vgg19, VGG19_Weights  # type: ignore

from data_loader import get_dataloader  # type: ignore
from models.generator import Generator  # type: ignore
from models.discriminator import Discriminator  # type: ignore


class VGGPerceptualLoss(nn.Module):  # type: ignore
    """Perceptual loss using VGG19 features for sharper, more realistic output."""

    def __init__(self, device):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        # Use layers up to relu3_3 for perceptual features
        vgg_layers = list(vgg.children())  # type: ignore
        self.features = nn.Sequential(*vgg_layers[:16]).eval().to(device)  # type: ignore
        for param in self.features.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

        # VGG normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def normalize(self, x):
        """Convert from [-1,1] to VGG input range."""
        x = (x + 1) / 2  # [-1,1] -> [0,1]
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        fake_features = self.features(self.normalize(fake))
        real_features = self.features(self.normalize(real))
        return self.criterion(fake_features, real_features)


def train():
    parser = argparse.ArgumentParser(description="Pix2Pix GAN Training")

    parser.add_argument('--dataset_path', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=10.0, help='Perceptual loss weight')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None, help='Path to generator checkpoint to resume from')

    opt = parser.parse_args()

    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ============================================================
    # Device Setup
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("  Pix2Pix GAN Training (with Perceptual Loss)")
    print("  Device: {}".format(device))
    if device.type == 'cuda':
        print("  GPU:    {}".format(torch.cuda.get_device_name(0)))
        print("  VRAM:   {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print("=" * 60)

    # ============================================================
    # Models
    # ============================================================
    generator = Generator(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=6).to(device)

   # Resume from checkpoint if specified
    if opt.resume and os.path.exists(opt.resume):
        generator.load_state_dict(torch.load(opt.resume, map_location=device, weights_only=True))
        print("  Resumed generator from: {}".format(opt.resume))

    # Try loading discriminator too
        d_path = opt.resume.replace("generator", "discriminator")
        if os.path.exists(d_path):
            discriminator.load_state_dict(torch.load(d_path, map_location=device, weights_only=True))
            print("  Resumed discriminator from: {}".format(d_path))

# ✅ ADD THIS HERE
    start_epoch = 0

    if opt.resume:
        try:
            start_epoch = int(opt.resume.split("epoch")[-1].split(".")[0])
            print(f"Resuming from epoch {start_epoch}")
        except:
            start_epoch = 0

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print("  Generator parameters:     {:,}".format(g_params))
    print("  Discriminator parameters: {:,}".format(d_params))
    print("=" * 60)

    # ============================================================
    # Loss Functions
    # ============================================================
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss(device)  # For sharper output

    # ============================================================
    # Optimizers
    # ============================================================
    optimizer_G = Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate schedulers - gradually reduce LR for finer details
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.5)

    # ============================================================
    # Data Loader
    # ============================================================
    dataloader = get_dataloader(opt.dataset_path, batch_size=opt.batch_size)
    total_batches = len(dataloader)

    # ============================================================
    # Training Loop
    # ============================================================
    best_g_loss = float('inf')
    print("\nStarting training: {} epochs, {} batches/epoch\n".format(opt.epochs, total_batches))

    for epoch in range(start_epoch, opt.epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_start = time.time()

        for i, batch in enumerate(dataloader):

            real_A = batch['occluded'].to(device)   # Masked face
            real_B = batch['real'].to(device)        # Unmasked face

            # Get patch shape dynamically
            with torch.no_grad():
                dummy_out = discriminator(real_B, real_A)
                patch_shape = dummy_out.shape

            valid = torch.ones(patch_shape, device=device) * 0.9
            fake = torch.zeros(patch_shape, device=device) + 0.1

            # ===========================
            #  Train Generator
            # ===========================
            optimizer_G.zero_grad()

            fake_B = generator(real_A)

            # Adversarial loss
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # L1 pixel loss
            loss_L1 = criterion_L1(fake_B, real_B)

            # Perceptual loss (VGG) - makes output SHARPER
            loss_perceptual = criterion_perceptual(fake_B, real_B)  # type: ignore

            # Total generator loss
            loss_G = loss_GAN + (opt.lambda_l1 * loss_L1) + (opt.lambda_perceptual * loss_perceptual)

            loss_G.backward()
            optimizer_G.step()

            # ===========================
            #  Train Discriminator
            # ===========================
            optimizer_D.zero_grad()

            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # Accumulate
            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()

            # Logging
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                print(
                    "  [Epoch {}/{}] [Batch {}/{}] "
                    "D: {:.4f}  G: {:.4f}  "
                    "(GAN: {:.4f}, L1: {:.4f}, VGG: {:.4f})".format(
                        epoch + 1, opt.epochs, i + 1, total_batches,
                        loss_D.item(), loss_G.item(),
                        loss_GAN.item(), loss_L1.item(), loss_perceptual.item()
                    )
                )

            # Save sample images
            if i % opt.sample_interval == 0:
                sample = torch.cat([
                    real_A[:4],
                    fake_B[:4],
                    real_B[:4],
                ], dim=0)
                vutils.save_image(
                    sample,
                    "results/epoch{:03d}_batch{:04d}.png".format(epoch + 1, i),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1)
                )

        # Step learning rate schedulers
        scheduler_G.step()
        scheduler_D.step()

        # Epoch Summary
        epoch_time = time.time() - epoch_start
        avg_d = epoch_d_loss / total_batches
        avg_g = epoch_g_loss / total_batches

        print("\n" + "-" * 60)
        print("  Epoch {}/{} completed in {:.1f}s | LR: {:.6f}".format(
            epoch + 1, opt.epochs, epoch_time, scheduler_G.get_last_lr()[0]))
        print("  Avg D_loss: {:.4f}  |  Avg G_loss: {:.4f}".format(avg_d, avg_g))
        print("-" * 60 + "\n")

        # Save checkpoints
        if (epoch + 1) % opt.save_interval == 0:
            torch.save(generator.state_dict(),
                       "saved_models/generator_epoch{:03d}.pth".format(epoch + 1))
            torch.save(discriminator.state_dict(),
                       "saved_models/discriminator_epoch{:03d}.pth".format(epoch + 1))
            print("  Checkpoint saved at epoch {}".format(epoch + 1))

        # Save best model
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            torch.save(generator.state_dict(), "saved_models/generator_best.pth")
            torch.save(discriminator.state_dict(), "saved_models/discriminator_best.pth")
            print("  Best model updated (G_loss: {:.4f})".format(avg_g))

    # Save final
    torch.save(generator.state_dict(), "saved_models/generator_final.pth")
    torch.save(discriminator.state_dict(), "saved_models/discriminator_final.pth")

    print("\n" + "=" * 60)
    print("  Training Completed!")
    print("  Best G_loss: {:.4f}".format(best_g_loss))
    print("  Models saved to: saved_models/")
    print("  Samples saved to: results/")
    print("=" * 60)


if __name__ == "__main__":
    train()