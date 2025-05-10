import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=[64, 128, 256, 512, 1024],
                 bilinear=True, use_attention=False, norm_type=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.filters = filters
        self.depth = len(filters) - 1

        self.inc = DoubleConv(n_channels, filters[0], norm_type=norm_type)
        self.downs = nn.ModuleList([
            Down(filters[i], filters[i+1], norm_type=norm_type)
            for i in range(self.depth)
        ])
        self.bottleneck = DoubleConv(filters[-1], filters[-1], norm_type=norm_type)

        self.ups = nn.ModuleList()
        in_channels = filters[-1]
        for i in range(self.depth):
            skip_channels = filters[-i-2]
            out_channels = skip_channels
            self.ups.append(
                Up(in_channels, skip_channels, out_channels,
                   bilinear=bilinear, use_attention=use_attention, norm_type=norm_type)
            )
            in_channels = out_channels

        self.outc = OutConv(filters[0], n_classes)


    def forward(self, x):
        # Encoder path
        encoder_features = []
        x = self.inc(x)
        encoder_features.append(x)
        for down in self.downs:
            x = down(x)
            encoder_features.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, up in enumerate(self.ups):
            skip = encoder_features[-i-2]  # Match with corresponding encoder feature
            x = up(x, skip)

        # Output
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.downs = nn.ModuleList([torch.utils.checkpoint(down) for down in self.downs])
        self.bottleneck = torch.utils.checkpoint(self.bottleneck)
        self.ups = nn.ModuleList([torch.utils.checkpoint(up) for up in self.ups])
        self.outc = torch.utils.checkpoint(self.outc)