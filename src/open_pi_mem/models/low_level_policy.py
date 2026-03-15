from __future__ import annotations

from typing import Any

import torch
from torch import nn

from open_pi_mem.models.action_expert import ActionExpert
from open_pi_mem.models.backbones import build_backbone_bundle
from open_pi_mem.models.video_memory import MEMVideoEncoder


class LowLevelPolicy(nn.Module):
    """Low-level policy for action execution.

    Architecture:
        - Text encoder: Gemma-style LM → L2 norm pool → (hidden_size)
        - Vision encoder: SigLIP + MEM temporal attention → (hidden_size)
        - Proprioception: action_dim projection → (hidden_size)
        - Fusion: concat + 2-layer MLP → (hidden_size)
        - Action expert: regress action trajectory (B, action_chunk_horizon, action_dim)
        - Optional FAST head: tokenize actions into discrete vocabulary
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        bundle = build_backbone_bundle(model_cfg)
        hidden_size = bundle.hidden_size  # typically 2304 for Gemma-2-2b
        action_dim = model_cfg["action_dim"]  # e.g., 14 for robot
        action_chunk_horizon = model_cfg["action_chunk_horizon"]  # e.g., 16 steps to predict

        # Text encoder: instruction -> hidden state
        self.text_backbone = bundle.text_backbone

        # Vision encoder: video frames with temporal attention (MEM style)
        self.video_encoder = MEMVideoEncoder(
            bundle.vision_tower,
            hidden_size=bundle.vision_tower.hidden_size,  # e.g., 768 for SigLIP
            temporal_every_n_layers=model_cfg.get("temporal_every_n_layers", 4),
            temporal_layers=model_cfg.get("temporal_layers", 2),
        )
        # Project vision features to shared hidden dimension
        self.video_projector = nn.Linear(
            bundle.vision_tower.hidden_size,  # e.g., 768
            hidden_size,  # e.g., 2304
        )

        # Project proprioception to shared hidden dimension
        self.proprio_proj = nn.Linear(action_dim, hidden_size)

        # Fusion network: concatenate all modalities then compress
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # cat(text, vision, proprio)
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Gradient control: whether to stop gradients from action expert to VLM backbone
        self.stop_action_expert_grad_to_backbone = model_cfg.get(
            "stop_action_expert_grad_to_backbone", True
        )

        # Optional FAST token head for discrete action tokenization
        self.fast_vocab_size = model_cfg.get("fast_vocab_size", 1024)
        self.fast_head = (
            nn.Linear(hidden_size, self.fast_vocab_size)
            if model_cfg.get("use_fast_head", True)
            else None
        )

        # Action expert: predict action trajectory
        self.action_expert = ActionExpert(
            hidden_size, action_dim, action_chunk_horizon
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        video: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for low-level policy with multi-modal fusion.

        Args:
            input_ids: (B, L) - tokenized instruction for current subtask
            video: (B, T, C, H, W) - video frames (T temporal, C channels, H×W spatial)
            proprio: (B, action_dim) - current proprioceptive state (joint angles/positions)
            attention_mask: (B, L) - mask for text tokens

        Returns:
            dict with:
                pooled_hidden: (B, hidden_size) - fused multimodal representation
                action_chunk: (B, action_chunk_horizon, action_dim) - predicted action trajectory
                fast_logits: (B, fast_vocab_size) - optional fast token logits (if use_fast_head=True)
        """
        # Text encoding: instruction -> hidden representation
        text_hidden = self.text_backbone.encode(
            input_ids,  # (B, L)
            attention_mask=attention_mask,  # (B, L)
        ).mean(dim=1)  # pool over sequence -> (B, hidden_size)

        # Vision encoding: video frames with temporal MEM attention -> hidden representation
        vision_hidden = self.video_projector(
            self.video_encoder(video).mean(dim=1)  # (B, T, num_patches, vis_hidden) -> (B, num_patches, vis_hidden) -> (B, vis_hidden)
        )  # (B, hidden_size)

        # Proprioception encoding: current state -> hidden representation
        proprio_hidden = self.proprio_proj(
            proprio.mean(dim=1)  # (B, action_dim) -> (B, action_dim)
        )  # (B, hidden_size)

        # Fuse all modalities: concatenate then MLP
        fused = self.fusion(
            torch.cat([text_hidden, vision_hidden, proprio_hidden], dim=-1)  # (B, 3*hidden_size)
        )  # (B, hidden_size)

        # Action expert: optionally detach gradients from backbone
        action_expert_input = (
            fused.detach() if self.stop_action_expert_grad_to_backbone else fused
        )  # (B, hidden_size)

        outputs = {
            "pooled_hidden": fused,  # (B, hidden_size)
            "action_chunk": self.action_expert(action_expert_input),  # (B, action_chunk_horizon, action_dim)
        }

        # Optional FAST token head for action tokenization
        if self.fast_head is not None:
            outputs["fast_logits"] = self.fast_head(fused)  # (B, fast_vocab_size)

        return outputs
