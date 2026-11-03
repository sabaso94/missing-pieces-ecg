"""A 1D ResNet for 12-lead ECGs, aware of which leads are present, with an optional tabular branch."""

import torch
from torch import nn

from ptbxl_reliable.leads import apply_mask


class ResidualBlock(nn.Module):
    """conv-BN-ReLU-conv-BN plus a shortcut. The shortcut is a 1x1 conv when the shape changes."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ECGResNet(nn.Module):
    """Backbone in the spirit of the PTB-XL benchmark's resnet1d_wang.

    Stem, then three stages with kernel sizes 7, 5 and 3, halving the time resolution between
    stages. Global average and max pooling make the model work for any recording length.
    The pooled ECG features are concatenated with the lead mask (so the head knows which leads
    were available) and, optionally, with an embedding of tabular metadata (late fusion).
    """

    def __init__(self, n_leads: int = 12, n_classes: int = 5, width: int = 128, n_tab: int = 0):
        super().__init__()
        self.n_tab = n_tab
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, width, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.stages = nn.Sequential(
            ResidualBlock(width, width, kernel_size=7),
            ResidualBlock(width, width, kernel_size=7),
            ResidualBlock(width, width, kernel_size=5, stride=2),
            ResidualBlock(width, width, kernel_size=5),
            ResidualBlock(width, width, kernel_size=3, stride=2),
            ResidualBlock(width, width, kernel_size=3),
        )

        head_in = 2 * width + n_leads
        if n_tab > 0:
            self.tab_mlp = nn.Sequential(nn.Linear(n_tab, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            head_in += 32
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def features(self, x: torch.Tensor, lead_mask: torch.Tensor) -> torch.Tensor:
        """Pooled ECG features (B, 2 * width): what the network learned from the signal itself.

        The lead mask and metadata are left out, so these are comparable across records and are
        what the shift tests in E4 look at.
        """
        h = self.stages(self.stem(apply_mask(x, lead_mask)))
        return torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)

    def forward(self, x: torch.Tensor, lead_mask: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        """Logits (B, n_classes). Absent leads are zeroed here, so callers can't forget to."""
        if self.n_tab > 0 and tab is None:
            raise ValueError("this model was built with a tabular branch and needs `tab`")

        features = [self.features(x, lead_mask), lead_mask.float()]
        if self.n_tab > 0:
            features.append(self.tab_mlp(tab))
        return self.head(torch.cat(features, dim=1))
