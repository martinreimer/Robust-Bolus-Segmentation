import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN|IN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        def norm_layer(channels):
            if norm_type == 'batch':
                return nn.BatchNorm2d(channels)
            elif norm_type == 'instance':
                return nn.InstanceNorm2d(channels, affine=True)
            else:
                return nn.Identity()  # No normalization

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionGate(nn.Module):
    def __init__(self, g_in_channels, x_in_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        #print(f"Use attention:\n g1: {g1.shape}, x1: {x1.shape}")
        psi = self.relu(g1 + x1)
        #print(f"psi: {psi.shape}")
        psi = self.psi(psi)
        #print(f"psi: {psi.shape}")
        return x * psi


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, use_attention=False, norm_type=None):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.channel_reduction = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)

        concat_channels = skip_channels + skip_channels
        self.conv = DoubleConv(concat_channels, out_channels, norm_type=norm_type)

        self.use_attention = use_attention
        if use_attention:
            self.attention_gate = AttentionGate(
                g_in_channels=in_channels,
                x_in_channels=in_channels,
                out_channels=out_channels
            )

    def forward(self, x, x_skip_con):
        x1 = self.up(x)
        if hasattr(self, 'channel_reduction'):
            x1 = self.channel_reduction(x1)
        if self.use_attention:
            x_skip_con = self.attention_gate(g=x1, x=x_skip_con)
        x = torch.cat([x_skip_con, x1], dim=1)
        return self.conv(x)


# Unchanged OutConv: Final 1x1 convolution
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
