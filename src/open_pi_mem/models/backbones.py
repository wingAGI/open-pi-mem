from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer


@dataclass
class BackboneBundle:
    text_backbone: "CausalTextBackbone"
    vision_tower: "VisionTower"
    hidden_size: int


class CausalTextBackbone(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        attn_implementation: str | None = None,
        dtype: str | None = None,
        freeze: bool = False,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        torch_dtype = _resolve_dtype(dtype)
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = int(self.model.config.hidden_size)
        if freeze:
            self.requires_grad_(False)

    def tokenize_texts(
        self,
        texts: list[str],
        *,
        max_length: int,
        device: torch.device | None = None,
        padding: bool = True,
    ) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="longest" if padding else False,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )


class VisionTower(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str | None = None,
        freeze: bool = False,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        torch_dtype = _resolve_dtype(dtype)
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        self.processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            vision_config = getattr(self.model.config, "vision_config", None)
            hidden_size = getattr(vision_config, "hidden_size", None)
        self.hidden_size = int(hidden_size)
        if freeze:
            self.requires_grad_(False)

    def preprocess_images(self, images: list[Any], *, device: torch.device | None = None) -> torch.Tensor:
        encoded = self.processor(images=images, return_tensors="pt")
        pixel_values = encoded["pixel_values"]
        if device is not None:
            pixel_values = pixel_values.to(device)
        return pixel_values

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("Vision tower did not return hidden states.")
        return hidden_states[-1]


def build_backbone_bundle(model_cfg: dict[str, Any]) -> BackboneBundle:
    text_backbone = CausalTextBackbone(
        model_name_or_path=model_cfg["backbone_name"],
        attn_implementation=model_cfg.get("attn_implementation"),
        dtype=model_cfg.get("torch_dtype"),
        freeze=model_cfg.get("freeze_text_backbone", False),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        local_files_only=model_cfg.get("local_files_only", False),
    )
    vision_tower = VisionTower(
        model_name_or_path=model_cfg["vision_tower_name"],
        dtype=model_cfg.get("torch_dtype"),
        freeze=model_cfg.get("freeze_vision_tower", False),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        local_files_only=model_cfg.get("local_files_only", False),
    )
    bundle = BackboneBundle(
        text_backbone=text_backbone,
        vision_tower=vision_tower,
        hidden_size=text_backbone.hidden_size,
    )
    load_optional_backbone_weights(bundle, model_cfg)
    return bundle


def load_optional_backbone_weights(bundle: BackboneBundle, model_cfg: dict[str, Any]) -> None:
    text_checkpoint = model_cfg.get("text_checkpoint")
    if text_checkpoint:
        state = _load_state_dict(text_checkpoint)
        bundle.text_backbone.load_state_dict(state, strict=False)
    vision_checkpoint = model_cfg.get("vision_checkpoint")
    if vision_checkpoint:
        state = _load_state_dict(vision_checkpoint)
        bundle.vision_tower.load_state_dict(state, strict=False)
    vlm_checkpoint = model_cfg.get("vlm_checkpoint")
    if vlm_checkpoint:
        state = _load_state_dict(vlm_checkpoint)
        text_state = _filter_state_dict(state, model_cfg.get("vlm_text_prefixes", ["text_backbone.", "language_model.", "model."]))
        vision_state = _filter_state_dict(state, model_cfg.get("vlm_vision_prefixes", ["vision_tower.", "vision_model.", "vision_tower.model."]))
        if text_state:
            bundle.text_backbone.load_state_dict(text_state, strict=False)
        if vision_state:
            bundle.vision_tower.load_state_dict(vision_state, strict=False)


def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
    if not dtype:
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".safetensors":
        return load_safetensors(str(checkpoint_path))
    obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def _filter_state_dict(state: dict[str, torch.Tensor], prefixes: list[str]) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                filtered[key[len(prefix) :]] = value
                break
    return filtered
