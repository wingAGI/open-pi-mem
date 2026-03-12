from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW

from open_pi_mem.models.high_level_policy import HighLevelPolicy


@dataclass
class HighLevelBatch:
    model_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor


class HighLevelTrainer:
    def __init__(self, model: HighLevelPolicy, learning_rate: float = 2e-5, weight_decay: float = 0.01) -> None:
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_step(self, batch: HighLevelBatch) -> dict[str, float]:
        self.model.train()
        outputs = self.model(
            labels=batch.labels,
            **batch.model_inputs,
        )
        loss = outputs["loss"]
        if loss is None:
            raise ValueError("High-level policy did not produce a language-model loss.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu())}
