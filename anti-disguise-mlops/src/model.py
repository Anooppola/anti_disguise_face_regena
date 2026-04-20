"""
model.py - Pix2Pix GAN Architecture
Implements U-Net Generator + PatchGAN Discriminator + VGG19 Perceptual Loss
"""

import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  U-Net Building Blocks
# ─────────────────────────────────────────────

class UNetDown(nn.Module):
    """Encoder block: Conv → InstanceNorm → LeakyReLU"""

    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose → InstanceNorm → ReLU + skip connection"""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.model(x), skip], dim=1)


# ─────────────────────────────────────────────
#  Generator (U-Net 256x256)
# ─────────────────────────────────────────────

class Generator(nn.Module):
    """
    U-Net Generator for Pix2Pix.
    Input:  (B, 3, 256, 256) masked face
    Output: (B, 3, 256, 256) reconstructed face
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.d1 = UNetDown(in_channels, 64, normalize=False)      # 256→128
        self.d2 = UNetDown(64, 128)                                # 128→64
        self.d3 = UNetDown(128, 256)                               # 64→32
        self.d4 = UNetDown(256, 512, dropout=0.5)                  # 32→16
        self.d5 = UNetDown(512, 512, dropout=0.5)                  # 16→8
        self.d6 = UNetDown(512, 512, dropout=0.5)                  # 8→4
        self.d7 = UNetDown(512, 512, dropout=0.5)                  # 4→2
        self.d8 = UNetDown(512, 512, normalize=False, dropout=0.5) # 2→1 (bottleneck)

        # Decoder
        self.u1 = UNetUp(512, 512, dropout=0.5)    # 1→2,  +d7 → 1024
        self.u2 = UNetUp(1024, 512, dropout=0.5)   # 2→4,  +d6 → 1024
        self.u3 = UNetUp(1024, 512, dropout=0.5)   # 4→8,  +d5 → 1024
        self.u4 = UNetUp(1024, 512)                # 8→16, +d4 → 1024
        self.u5 = UNetUp(1024, 256)                # 16→32,+d3 → 512
        self.u6 = UNetUp(512, 128)                 # 32→64,+d2 → 256
        self.u7 = UNetUp(256, 64)                  # 64→128,+d1 → 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),  # 128→256
            nn.Tanh()
        )

        self.apply(self._init_weights)
        logger.info("Generator initialized (%.2fM params)", self._count_params() / 1e6)

    def _count_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init_weights(m):
        cls = m.__class__.__name__
        if "Conv" in cls:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif "Norm" in cls and hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)

        return self.final(u7)


# ─────────────────────────────────────────────
#  Discriminator (PatchGAN 70×70)
# ─────────────────────────────────────────────

class PatchGANBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True, stride=2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Classifies 70×70 overlapping patches as real or fake.
    Input:  concat(masked_image, target_image) → (B, 6, 256, 256)
    Output: (B, 1, 30, 30) patch predictions
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            PatchGANBlock(in_channels * 2, 64, normalize=False),  # 256→128
            PatchGANBlock(64, 128),                                # 128→64
            PatchGANBlock(128, 256),                               # 64→32
            PatchGANBlock(256, 512, stride=1),                     # 32→31
            nn.Conv2d(512, 1, 4, 1, 1),                            # 31→30
        )

        self.apply(self._init_weights)
        logger.info("Discriminator initialized (%.2fM params)", self._count_params() / 1e6)

    def _count_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init_weights(m):
        cls = m.__class__.__name__
        if "Conv" in cls:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif "Norm" in cls and hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, masked, target):
        x = torch.cat([masked, target], dim=1)
        return self.model(x)


# ─────────────────────────────────────────────
#  VGG19 Perceptual Loss
# ─────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Computes feature-level L1 loss between generated and real images
    using pre-trained VGG19 features (relu2_2 layer).
    """

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # Use layers up to relu2_2 (index 9)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:10]).to(device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # Normalize from [-1,1] to ImageNet range
        mean = torch.tensor([0.485, 0.456, 0.406], device=generated.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=generated.device).view(1, 3, 1, 1)
        gen_norm  = (generated * 0.5 + 0.5 - mean) / std
        real_norm = (real      * 0.5 + 0.5 - mean) / std
        return self.criterion(self.feature_extractor(gen_norm),
                              self.feature_extractor(real_norm))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    D = Discriminator().to(device)
    x = torch.randn(1, 3, 256, 256, device=device)
    out_g = G(x)
    out_d = D(x, out_g)
    print(f"Generator  output: {out_g.shape}")
    print(f"Discriminator output: {out_d.shape}")
