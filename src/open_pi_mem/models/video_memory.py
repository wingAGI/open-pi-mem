from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, frames, patches, hidden]
        b, t, p, d = x.shape
        tokens = rearrange(x, "b t p d -> (b p) t d")
        causal_mask = torch.full((t, t), float("-inf"), device=x.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attended, _ = self.temporal_attn(tokens, tokens, tokens, attn_mask=causal_mask)
        tokens = self.norm(tokens + attended)
        tokens = tokens + self.ff(tokens)
        return rearrange(tokens, "(b p) t d -> b t p d", b=b, p=p)


class MEMVideoEncoder(nn.Module):
    def __init__(self, vision_tower: nn.Module, hidden_size: int, temporal_every_n_layers: int = 4, temporal_layers: int = 2) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.hidden_size = hidden_size
        self.temporal_every_n_layers = temporal_every_n_layers
        self.temporal_layers = nn.ModuleList([TemporalBlock(hidden_size) for _ in range(temporal_layers)])

    def temporal_position(self, frames: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(frames, device=device).float()
        pos = pos - (frames - 1)
        dim = self.hidden_size
        freqs = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        emb = torch.zeros(frames, dim, device=device)
        emb[:, 0::2] = torch.sin(pos[:, None] * freqs[None, :])
        emb[:, 1::2] = torch.cos(pos[:, None] * freqs[None, :])
        emb[-1] = 0.0  # enforce e(0)=0 for current frame compatibility
        return emb

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: [batch, frames, channels, height, width]
        b, t, c, h, w = video.shape
        flat = rearrange(video, "b t c h w -> (b t) c h w")
        patch_tokens = self.vision_tower(flat)
        patch_tokens = rearrange(patch_tokens, "(b t) p d -> b t p d", b=b, t=t)
        patch_tokens = patch_tokens + self.temporal_position(t, patch_tokens.device)[None, :, None, :]
        for block in self.temporal_layers:
            patch_tokens = block(patch_tokens)
        # Keep only current-frame tokens for downstream backbone, matching MEM's compression idea.
        return patch_tokens[:, -1]
