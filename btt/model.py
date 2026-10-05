"""A deliberately small CRNN for delivery (emotion) classification.

Log-mel (1, 64, 300) -> 3 conv blocks -> (GRU over time) -> mean pool -> linear.
Feature standardisation lives inside the module as buffers, so the exported
ONNX file takes raw log-mel and cannot drift from training.
"""
from __future__ import annotations

import torch
from torch import nn

from .features import N_MELS, WIN_FRAMES
from .labels import CLASSES


def _block(c_in: int, c_out: int, pool: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class DeliveryNet(nn.Module):
    def __init__(
        self,
        n_classes: int = len(CLASSES),
        channels: tuple[int, int, int] = (16, 32, 64),
        gru_hidden: int = 64,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.register_buffer("feat_mean", torch.zeros(1, 1, 1, 1))
        self.register_buffer("feat_std", torch.ones(1, 1, 1, 1))
        self.conv = nn.Sequential(
            _block(1, c1, (2, 2)),  # 64x300 -> 32x150
            _block(c1, c2, (2, 2)),  # -> 16x75
            _block(c2, c3, (2, 1)),  # -> 8x75
        )
        self.proj = nn.Linear(c3 * (N_MELS // 8), 64)
        self.gru = nn.GRU(64, gru_hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = gru_hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, n_classes)
        self.bidirectional = bidirectional
        self.channels = channels
        self.gru_hidden = gru_hidden

    def set_norm(self, mean: float, std: float) -> None:
        self.feat_mean.fill_(float(mean))
        self.feat_std.fill_(float(max(std, 1e-6)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, N_MELS, T) raw log-mel -> logits (B, n_classes)."""
        x = (x - self.feat_mean) / self.feat_std
        x = self.conv(x)  # (B, C, F, T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F)
        x = torch.relu(self.proj(x))
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.head(self.drop(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def example_input(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, 1, N_MELS, WIN_FRAMES)
