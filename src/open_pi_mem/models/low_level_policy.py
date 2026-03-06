from __future__ import annotations

from typing import Any

import torch
from torch import nn

from open_pi_mem.models.action_expert import ActionExpert
from open_pi_mem.models.backbones import build_backbone_bundle
from open_pi_mem.models.video_memory import MEMVideoEncoder


class LowLevelPolicy(nn.Module):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        bundle = build_backbone_bundle(model_cfg)
        hidden_size = bundle.hidden_size
        action_dim = model_cfg["action_dim"]
        action_chunk_horizon = model_cfg["action_chunk_horizon"]
        self.text_backbone = bundle.text_backbone
        self.video_encoder = MEMVideoEncoder(
            bundle.vision_tower,
            hidden_size=bundle.vision_tower.hidden_size,
            temporal_every_n_layers=model_cfg.get("temporal_every_n_layers", 4),
            temporal_layers=model_cfg.get("temporal_layers", 2),
        )
        self.video_projector = nn.Linear(bundle.vision_tower.hidden_size, hidden_size)
        self.proprio_proj = nn.Linear(action_dim, hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.stop_action_expert_grad_to_backbone = model_cfg.get("stop_action_expert_grad_to_backbone", True)
        self.fast_vocab_size = model_cfg.get("fast_vocab_size", 1024)
        self.fast_head = nn.Linear(hidden_size, self.fast_vocab_size) if model_cfg.get("use_fast_head", True) else None
        self.action_expert = ActionExpert(hidden_size, action_dim, action_chunk_horizon)

    def forward(
        self,
        input_ids: torch.Tensor,
        video: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_hidden = self.text_backbone.encode(input_ids, attention_mask=attention_mask).mean(dim=1)
        vision_hidden = self.video_projector(self.video_encoder(video).mean(dim=1))
        proprio_hidden = self.proprio_proj(proprio.mean(dim=1))
        fused = self.fusion(torch.cat([text_hidden, vision_hidden, proprio_hidden], dim=-1))
        action_expert_input = fused.detach() if self.stop_action_expert_grad_to_backbone else fused
        outputs = {
            "pooled_hidden": fused,
            "action_chunk": self.action_expert(action_expert_input),
        }
        if self.fast_head is not None:
            outputs["fast_logits"] = self.fast_head(fused)
        return outputs
