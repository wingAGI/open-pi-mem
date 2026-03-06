from __future__ import annotations

import torch
from torch import nn


class ActionExpert(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim * horizon),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        actions = self.mlp(pooled_hidden)
        return actions.view(actions.shape[0], self.horizon, self.action_dim)
