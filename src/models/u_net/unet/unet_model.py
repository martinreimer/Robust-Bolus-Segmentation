import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *


# Updated UNet: Parameterized and flexible
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=[64, 128, 256, 512, 1024], bilinear=True, use_attention=False):
        """
        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output classes
            filters (list): List of filter sizes for each encoder level (e.g., [64, 128, 256, 512, 1024])
            bilinear (bool): Use bilinear upsampling if True, else use transposed convolution
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.filters = filters
        self.depth = len(filters) - 1  # Number of down/up steps

        # Encoder
        self.inc = DoubleConv(n_channels, filters[0])
        self.downs = nn.ModuleList([Down(filters[i], filters[i+1]) for i in range(self.depth)])

        # Bottleneck
        self.bottleneck = DoubleConv(filters[-1], filters[-1])

        # Decoder
        self.ups = nn.ModuleList()
        in_channels = filters[-1]  # Start with bottleneck channels
        for i in range(self.depth):
            skip_channels = filters[-i-2]  # Encoder feature channels
            out_channels = skip_channels  # Match encoder level
            self.ups.append(Up(in_channels, skip_channels, out_channels, bilinear, use_attention))
            in_channels = out_channels  # Next layer’s input is this layer’s output

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