import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix.
    Input: concatenation of (generated/real image, condition image) → 6 channels
    Output: patch-level predictions (B, 1, 30, 30) for 256×256 input
    """

    def __init__(self, in_channels=6):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Standard discriminator block: Conv → Norm → LeakyReLU"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),  # 256 → 128
            *discriminator_block(64, 128),                               # 128 → 64
            *discriminator_block(128, 256),                              # 64  → 32
            *discriminator_block(256, 512),                              # 32  → 16
            nn.ZeroPad2d((1, 0, 1, 0)),                                  # 16  → 17
            nn.Conv2d(512, 1, 4, padding=1, bias=False)                  # 17  → 16
        )

        # Apply weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Initialize weights with normal distribution (mean=0, std=0.02)."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Norm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, img_A, img_B):
        """
        Forward pass.
        img_A: generated/real image (B, 3, 256, 256)
        img_B: condition image (B, 3, 256, 256) — the masked input
        """
        img_input = torch.cat((img_A, img_B), dim=1)  # (B, 6, 256, 256)
        return self.model(img_input)


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    d = Discriminator()
    out = d(x, y)
    print(f"Input shapes:  {x.shape}, {y.shape}")
    print(f"Output shape:  {out.shape}")
    print(f"Parameters:    {sum(p.numel() for p in d.parameters()):,}")
