from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW

from open_pi_mem.models.low_level_policy import LowLevelPolicy
from open_pi_mem.training.losses import action_chunk_loss, fast_token_loss, flow_matching_loss


@dataclass
class LowLevelBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    video: torch.Tensor
    proprio: torch.Tensor
    target_actions: torch.Tensor
    fast_targets: torch.Tensor | None = None


class LowLevelTrainer:
    def __init__(
        self,
        model: LowLevelPolicy,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        action_mse_weight: float = 1.0,
        fast_token_weight: float = 1.0,
        flow_matching_weight: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.action_mse_weight = action_mse_weight
        self.fast_token_weight = fast_token_weight
        self.flow_matching_weight = flow_matching_weight

    def train_step(self, batch: LowLevelBatch) -> dict[str, float]:
        self.model.train()
        outputs = self.model(batch.input_ids, batch.video, batch.proprio, attention_mask=batch.attention_mask)
        mse = action_chunk_loss(outputs["action_chunk"], batch.target_actions)
        flow = flow_matching_loss(outputs["action_chunk"], batch.target_actions)
        total = self.action_mse_weight * mse + self.flow_matching_weight * flow
        fast = None
        if batch.fast_targets is not None and "fast_logits" in outputs:
            fast = fast_token_loss(outputs["fast_logits"], batch.fast_targets)
            total = total + self.fast_token_weight * fast
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        metrics = {
            "loss": float(total.detach().cpu()),
            "action_mse": float(mse.detach().cpu()),
            "flow_loss": float(flow.detach().cpu()),
        }
        if fast is not None:
            metrics["fast_loss"] = float(fast.detach().cpu())
        return metrics
