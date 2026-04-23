# Vendored from https://github.com/minzwon/musicfm
# MIT License - Copyright 2023 ByteDance Inc. (see LICENSE in this directory)
from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, einsum, nn


class RandomProjectionQuantizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_dim: int,
        codebook_size: int,
        seed: int = 142,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        random_projection = torch.empty(input_dim, codebook_dim)
        nn.init.xavier_normal_(random_projection)
        self.register_buffer("random_projection", random_projection)
        codebook = torch.empty(codebook_size, codebook_dim)
        nn.init.normal_(codebook)
        self.register_buffer("codebook", codebook)

    def codebook_lookup(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        x = rearrange(x, "b n e -> (b n) e")
        normalized_x = nn.functional.normalize(x, dim=1, p=2)
        codebook: Tensor = self.codebook  # type: ignore[assignment]
        normalized_codebook = nn.functional.normalize(codebook, dim=1, p=2)
        distances = torch.cdist(normalized_codebook, normalized_x)
        nearest_indices = torch.argmin(distances, dim=0)
        return rearrange(nearest_indices, "(b n) -> b n", b=b)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        self.eval()
        x = einsum("b n d, d e -> b n e", x, self.random_projection)
        return self.codebook_lookup(x)
