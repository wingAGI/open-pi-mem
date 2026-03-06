from __future__ import annotations

from typing import Any

import torch
from torch import nn

from open_pi_mem.models.backbones import build_backbone_bundle


class HighLevelPolicy(nn.Module):
    """Train the high-level policy as a causal LM over structured outputs.

    Target strings should encode both the next subtask and next memory summary.
    Example:
    <subtask>pick up mug</subtask>\n<memory>mug moved from sink to counter</memory>
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        bundle = build_backbone_bundle(model_cfg)
        self.text_backbone = bundle.text_backbone
        self.vision_tower = bundle.vision_tower
        self.hidden_size = bundle.hidden_size
        self.vision_projector = nn.Linear(bundle.vision_tower.hidden_size, self.hidden_size)

    def encode_visual_context(self, pixel_values: torch.Tensor | None) -> torch.Tensor | None:
        if pixel_values is None:
            return None
        tokens = self.vision_tower(pixel_values)
        pooled = tokens.mean(dim=1)
        return self.vision_projector(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        hidden = outputs.hidden_states[-1]
        result = {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": hidden,
        }
        vision_context = self.encode_visual_context(pixel_values)
        if vision_context is not None:
            result["vision_context"] = vision_context
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        return self.text_backbone.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
