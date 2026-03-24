import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """Downsampling block for U-Net encoder."""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block for U-Net decoder with skip connections."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        return x


class Generator(nn.Module):
    """
    U-Net Generator for Pix2Pix.
    Input:  (B, 3, 256, 256) masked face
    Output: (B, 3, 256, 256) reconstructed face
    
    Architecture: 8 downsampling → 7 upsampling + skip connections → final layer
    256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1 (bottleneck)
    1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → 256 (output)
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)     # 256 → 128
        self.down2 = UNetDown(64, 128)                               # 128 → 64
        self.down3 = UNetDown(128, 256)                              # 64  → 32
        self.down4 = UNetDown(256, 512, dropout=0.5)                 # 32  → 16
        self.down5 = UNetDown(512, 512, dropout=0.5)                 # 16  → 8
        self.down6 = UNetDown(512, 512, dropout=0.5)                 # 8   → 4
        self.down7 = UNetDown(512, 512, dropout=0.5)                 # 4   → 2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)  # 2 → 1 (bottleneck)

        # Decoder (upsampling with skip connections)
        self.up1 = UNetUp(512, 512, dropout=0.5)     # 1 → 2,  concat with down7 → 1024
        self.up2 = UNetUp(1024, 512, dropout=0.5)    # 2 → 4,  concat with down6 → 1024
        self.up3 = UNetUp(1024, 512, dropout=0.5)    # 4 → 8,  concat with down5 → 1024
        self.up4 = UNetUp(1024, 512)                 # 8 → 16, concat with down4 → 1024
        self.up5 = UNetUp(1024, 256)                 # 16 → 32, concat with down3 → 512
        self.up6 = UNetUp(512, 128)                  # 32 → 64, concat with down2 → 256
        self.up7 = UNetUp(256, 64)                   # 64 → 128, concat with down1 → 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),  # 128 → 256
            nn.Tanh()
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

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)  # Bottleneck

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    g = Generator()
    out = g(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in g.parameters()):,}")
