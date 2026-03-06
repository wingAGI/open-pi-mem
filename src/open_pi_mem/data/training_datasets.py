from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from open_pi_mem.data.schemas import LowLevelTrainingRecord, MemoryTrainingRecord
from open_pi_mem.utils.io import read_jsonl


IGNORE_INDEX = -100


@dataclass
class HighLevelCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompts = [row["prompt"] for row in rows]
        full_texts = [f"{row['prompt']}{row['target']}" for row in rows]
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        for idx in range(labels.shape[0]):
            prompt_len = int(prompt_tokens["attention_mask"][idx].sum().item())
            labels[idx, :prompt_len] = IGNORE_INDEX
        encoded["labels"] = labels
        return encoded


class MemorySupervisionDataset(Dataset[dict[str, str]]):
    def __init__(self, jsonl_path: str | Path) -> None:
        self.rows = [MemoryTrainingRecord.model_validate(row) for row in read_jsonl(jsonl_path)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        history_text = "\n".join(
            f"- {item['text']} | status={item['status']}" for item in row.history
        )
        prompt = (
            "You are the high-level planner for a robot policy.\n"
            f"Goal: {row.goal}\n"
            f"Previous memory: {row.prev_memory or 'None'}\n"
            f"History:\n{history_text}\n"
            "Predict the next subtask and memory update.\n"
        )
        target = (
            f"<subtask>{row.next_subtask}</subtask>\n"
            f"<memory>{row.next_memory}</memory>"
        )
        return {"prompt": prompt, "target": target}


@dataclass
class LowLevelCollator:
    tokenizer: Any
    image_processor: Any
    max_prompt_length: int
    max_frames: int
    action_dim: int

    def __call__(self, rows: list[LowLevelTrainingRecord]) -> dict[str, torch.Tensor]:
        prompts = [f"Goal: {row.goal}\nSubtask: {row.subtask}" for row in rows]
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        video_batch = []
        proprio_batch = []
        action_batch = []
        fast_batch = []
        for row in rows:
            images = [_load_image(path) for path in row.frame_paths[: self.max_frames]]
            encoded = self.image_processor(images=images, return_tensors="pt")
            pixel_values = encoded["pixel_values"]
            video_batch.append(_pad_frames(pixel_values, self.max_frames))
            proprio = torch.tensor(row.proprio[: self.max_frames], dtype=torch.float32)
            proprio_batch.append(_pad_timesteps(proprio, self.max_frames, self.action_dim))
            actions = torch.tensor(row.action_chunk, dtype=torch.float32)
            action_batch.append(actions)
            if row.fast_tokens is not None and len(row.fast_tokens) > 0:
                fast_batch.append(int(row.fast_tokens[0]))
            else:
                fast_batch.append(0)
        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "video": torch.stack(video_batch, dim=0),
            "proprio": torch.stack(proprio_batch, dim=0),
            "target_actions": torch.stack(action_batch, dim=0),
            "fast_targets": torch.tensor(fast_batch, dtype=torch.long),
        }
        return batch


class JsonlLowLevelDataset(Dataset[LowLevelTrainingRecord]):
    def __init__(self, jsonl_path: str | Path) -> None:
        self.rows = [LowLevelTrainingRecord.model_validate(row) for row in read_jsonl(jsonl_path)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> LowLevelTrainingRecord:
        return self.rows[index]


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pad_frames(pixel_values: torch.Tensor, max_frames: int) -> torch.Tensor:
    if pixel_values.shape[0] >= max_frames:
        return pixel_values[:max_frames]
    pad = pixel_values[-1:].repeat(max_frames - pixel_values.shape[0], 1, 1, 1)
    return torch.cat([pixel_values, pad], dim=0)


def _pad_timesteps(values: torch.Tensor, max_frames: int, feature_dim: int) -> torch.Tensor:
    if values.shape[0] >= max_frames:
        return values[:max_frames]
    if values.numel() == 0:
        values = torch.zeros(1, feature_dim, dtype=torch.float32)
    pad = values[-1:].repeat(max_frames - values.shape[0], 1)
    return torch.cat([values, pad], dim=0)
