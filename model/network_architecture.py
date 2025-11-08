import torch
import torch.nn as nn
from torchvision import models

# custom weight init from your project
from model.initial_weight import weights_initialize_normal as weights_initialize


# ---------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------
class ResidualLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------
# Down-sampling residual block
# ---------------------------------------------------------------------
class DownResidualLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # residual down-sampling branch
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # stride=2: H, W -> 1/2
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # keep size
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # residual part after down-sampling
        self.residual = ResidualLayer(out_channels, out_channels)

        # skip connection to match channels & resolution
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.main(x)
        skip_out = self.skip(x)
        fused = main_out + skip_out
        out = self.residual(fused) + fused
        return out


# ---------------------------------------------------------------------
# Up-sampling residual block
# ---------------------------------------------------------------------
class UpResidualLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # main up-sampling branch
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # upsample by 2
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # keep size
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # extra residual block
        self.residual = ResidualLayer(out_channels, out_channels)

        # skip up-sampling
        self.skip = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.main(x)
        skip_out = self.skip(x)
        fused = main_out + skip_out
        out = self.residual(fused) + fused
        return out


# ---------------------------------------------------------------------
# Dilated residual layer
# ---------------------------------------------------------------------
class DilatedResidualLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
        )

        self.stage2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage1(x) + x
        out = self.stage2(out) + out
        return out

# ---------------------------------------------------------------------
# Phase Generator
# ---------------------------------------------------------------------
class Phase_Generator(nn.Module):

    def __init__(self, num_in: int, num_out: int):
        super(Phase_Generator, self).__init__()

        self.input_channel = num_in
        self.output_channel = num_out

        # small dropout for regularization
        self.dropout = nn.Dropout(p=0.02)

        # encoder / down
        self.down1 = DownResidualLayer(self.input_channel, 16)
        self.down2 = DownResidualLayer(16, 32)
        self.down3 = DownResidualLayer(32, 48)
        self.down4 = DownResidualLayer(48, 96)
        self.down5 = DownResidualLayer(96, 128)

        # decoder / up
        self.up1 = UpResidualLayer(128, 192)
        self.up2 = UpResidualLayer(192 + 96, 96)
        self.up3 = UpResidualLayer(96 + 48, 48)
        self.up4 = UpResidualLayer(48 + 32, 32)
        self.up5 = UpResidualLayer(32 + 16, 32)

        # final refinement
        self.residual = ResidualLayer(32, 16)
        self.activation = nn.Sigmoid()
        self.conv_phase = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

        # initialize weight
        self.apply(weights_initialize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        x1 = self.dropout(self.down1(x))
        x2 = self.dropout(self.down2(x1))
        x3 = self.dropout(self.down3(x2))
        x4 = self.dropout(self.down4(x3))
        x5 = self.dropout(self.down5(x4))

        x6 = self.dropout(self.up1(x5))
        x6_cat = torch.cat([x4, x6], dim=1)

        x7 = self.dropout(self.up2(x6_cat))
        x7_cat = torch.cat([x3, x7], dim=1)

        x8 = self.dropout(self.up3(x7_cat))
        x8_cat = torch.cat([x2, x8], dim=1)

        x9 = self.dropout(self.up4(x8_cat))
        x9_cat = torch.cat([x1, x9], dim=1)

        x10 = self.up5(x9_cat)
        x11 = self.residual(x10)

        phase = self.conv_phase(self.activation(x11))
        return phase


# ---------------------------------------------------------------------
# Aberration Generator (ResNet34 backbone)
# ---------------------------------------------------------------------
class Aberration_Generator(nn.Module):

    def __init__(self, out_class: int):
        super(Aberration_Generator, self).__init__()
        self.resnet34 = models.resnet34()

        # adapt first conv to 1-channel input
        self.resnet34.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # replace final FC
        fc_in = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(fc_in, out_class)

        # weight init
        self.apply(weights_initialize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet34(x)


# ---------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    Simple CNN discriminator:
      Conv -> BN -> LReLU stacks + global pooling + 1x1 conv
    Returns score per sample.
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            self._disc_block(in_channels, hidden_dim),
            self._disc_block(hidden_dim, hidden_dim * 2),
            self._disc_block(hidden_dim * 2, hidden_dim * 4),
            self._disc_block(hidden_dim * 4, hidden_dim * 8),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=1, stride=1),
        )

    @staticmethod
    def _disc_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.model(x)
        return scores.view(len(scores), -1)
