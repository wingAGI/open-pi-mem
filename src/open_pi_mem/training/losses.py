from __future__ import annotations

import torch
import torch.nn.functional as F


def language_head_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def action_chunk_loss(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_actions, target_actions)


def fast_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def flow_matching_loss(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred_actions, target_actions)
