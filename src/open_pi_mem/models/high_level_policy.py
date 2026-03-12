from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from open_pi_mem.models.backbones import build_backbone_bundle


class HighLevelPolicy(nn.Module):
    """Train the high-level policy as a causal LM over structured outputs.

    Target strings should encode both the next subtask and next memory summary.
    Example:
    <subtask>pick up mug</subtask>\n<memory>mug moved from sink to counter</memory>
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.multimodal_backbone_name = model_cfg.get("multimodal_backbone_name")
        self.is_multimodal = self.multimodal_backbone_name is not None
        self.processor = None
        self.text_backbone = None
        self.vision_tower = None
        self.vision_projector = None
        self.hidden_size = 0
        self.model = None
        if self.is_multimodal:
            self._init_multimodal(model_cfg)
            return
        bundle = build_backbone_bundle(model_cfg)
        self.text_backbone = bundle.text_backbone
        self.vision_tower = bundle.vision_tower
        self.hidden_size = bundle.hidden_size
        self.vision_projector = nn.Linear(bundle.vision_tower.hidden_size, self.hidden_size)

    def _init_multimodal(self, model_cfg: dict[str, Any]) -> None:
        kwargs = {
            "trust_remote_code": model_cfg.get("trust_remote_code", False),
            "local_files_only": model_cfg.get("local_files_only", False),
        }
        torch_dtype = model_cfg.get("torch_dtype")
        if torch_dtype == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif torch_dtype in {"fp16", "float16"}:
            kwargs["torch_dtype"] = torch.float16
        elif torch_dtype in {"fp32", "float32"}:
            kwargs["torch_dtype"] = torch.float32
        attn_implementation = model_cfg.get("attn_implementation")
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.multimodal_backbone_name,
            **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.multimodal_backbone_name,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            local_files_only=model_cfg.get("local_files_only", False),
        )
        text_config = getattr(self.model.config, "text_config", None)
        self.hidden_size = int(getattr(text_config, "hidden_size", getattr(self.model.config, "hidden_size", 0)))
        if model_cfg.get("freeze_text_backbone", False):
            self.model.requires_grad_(False)

    def encode_visual_context(self, pixel_values: torch.Tensor | None) -> torch.Tensor | None:
        if self.is_multimodal:
            return None
        if pixel_values is None:
            return None
        tokens = self.vision_tower(pixel_values)
        pooled = tokens.mean(dim=1)
        return self.vision_projector(pooled)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **extra_inputs: Any,
    ) -> dict[str, torch.Tensor]:
        if self.is_multimodal:
            if self.model is None:
                raise ValueError("Multimodal model is not initialized.")
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                **extra_inputs,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            hidden = hidden_states[-1] if hidden_states else outputs.logits
            return {
                "loss": outputs.loss,
                "logits": outputs.logits,
                "hidden_states": hidden,
            }
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
        pixel_values: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        if self.is_multimodal:
            if self.model is None:
                raise ValueError("Multimodal model is not initialized.")
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **generate_kwargs,
            )
        return self.text_backbone.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
