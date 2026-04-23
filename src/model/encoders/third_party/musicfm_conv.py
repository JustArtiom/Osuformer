# Vendored from https://github.com/minzwon/musicfm
# MIT License - Copyright 2023 ByteDance Inc. (see LICENSE in this directory)
from __future__ import annotations

from einops import rearrange
from torch import Tensor, nn


class Res2dModule(nn.Module):
    def __init__(self, idim: int, odim: int, stride: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(idim, odim, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(odim)
        self.conv2 = nn.Conv2d(odim, odim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(odim)
        self.relu = nn.ReLU()
        self.diff = False
        if (idim != odim) or (stride[0] > 1):
            self.conv3 = nn.Conv2d(idim, odim, 3, padding=1, stride=stride)
            self.bn3 = nn.BatchNorm2d(odim)
            self.diff = True

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.diff:
            x = self.bn3(self.conv3(x))
        out = x + out
        return self.relu(out)


class Conv2dSubsampling(nn.Module):
    def __init__(
        self,
        idim: int,
        hdim: int,
        odim: int,
        strides: list[int] | tuple[int, int] = (2, 2),
        n_bands: int = 64,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            Res2dModule(idim, hdim, (2, strides[0])),
            Res2dModule(hdim, hdim, (2, strides[1])),
        )
        self.linear = nn.Linear(hdim * n_bands // 2 // 2, odim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = rearrange(x, "b c f t -> b t (c f)")
        x = self.linear(x)
        return x
