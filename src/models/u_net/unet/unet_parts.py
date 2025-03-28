import torch
import torch.nn as nn
import torch.nn.functional as F

# Unchanged DoubleConv: Two Conv2d -> BN -> ReLU layers
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Unchanged Down: MaxPool followed by DoubleConv
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
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
    """Upscaling then double conv"""
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, use_attention=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.channel_reduction = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
            concat_channels = skip_channels + skip_channels  # After concatenation
            self.conv = DoubleConv(concat_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
            concat_channels = skip_channels + skip_channels
            self.conv = DoubleConv(concat_channels, out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention_gate = AttentionGate(g_in_channels=in_channels, x_in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, x_skip_con):
        # Upsample
        x1 = self.up(x)
        # Reduce channels
        if hasattr(self, 'channel_reduction'):
            x1 = self.channel_reduction(x1)
        '''
        # Padding (instead of cropping the skip-con) to match skip connection size
        diffY = skip_x.size()[2] - x1.size()[2]
        diffX = skip_x.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        #print(f"After padding: x1: {x1.shape}, x2: {x2.shape}")
        '''
        # Attention gate
        if self.use_attention:
            x_skip_con = self.attention_gate(g=x1, x=x_skip_con)
            #print(f"After attention: x1: {x1.shape}, x2: {x2.shape}")

        #print(f"Concatenating: x1: {x1.shape}, x2: {x2.shape}")
        # Concat and apply conv
        x = torch.cat([x_skip_con, x1], dim=1)
        return self.conv(x)

# Unchanged OutConv: Final 1x1 convolution
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
