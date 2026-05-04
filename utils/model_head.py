"""Compact 1D U-Net head used for sequence-level prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers with the same output width."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """Downsample the sequence while preserving a skip connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        pooled = self.pool(x)
        return x, pooled


class Decoder(nn.Module):
    """Upsample and fuse a skip connection from the encoder path."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        upsample_channels = in_channels * 2 // 3
        self.upconv = nn.ConvTranspose1d(
            upsample_channels,
            upsample_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = F.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=256 + 2 * 128, out_channels=1):
        super().__init__()
        hidden_channels = in_channels * 2
        self.enc1 = Encoder(in_channels, hidden_channels)
        self.enc2 = Encoder(hidden_channels, hidden_channels * 2)

        self.bottleneck = ConvBlock(hidden_channels * 2, hidden_channels * 4)

        self.dec2 = Decoder(hidden_channels * 4 + hidden_channels * 2, hidden_channels * 2)
        self.dec1 = Decoder(hidden_channels * 2 + hidden_channels, hidden_channels)

        self.final_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Convert [batch, length, channels] to [batch, channels, length].
        x = x.permute(0, 2, 1)

        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)

        x = self.bottleneck(x)

        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = self.final_conv(x)

        return x
