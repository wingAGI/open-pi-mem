from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from open_pi_mem.data.training_datasets import JsonlLowLevelDataset, LowLevelCollator
from open_pi_mem.models.low_level_policy import LowLevelPolicy
from open_pi_mem.training.common import set_seed
from open_pi_mem.training.low_level_trainer import LowLevelBatch, LowLevelTrainer
from open_pi_mem.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 7))

    model = LowLevelPolicy(cfg["model"])
    trainer = LowLevelTrainer(
        model,
        learning_rate=cfg["trainer"]["learning_rate"],
        weight_decay=cfg["trainer"].get("weight_decay", 0.01),
        action_mse_weight=cfg["loss"]["action_mse_weight"],
        fast_token_weight=cfg["loss"]["fast_token_weight"],
        flow_matching_weight=cfg["loss"]["flow_matching_weight"],
    )
    dataset = JsonlLowLevelDataset(cfg["data"]["train_jsonl"])
    collator = LowLevelCollator(
        tokenizer=model.text_backbone.tokenizer,
        image_processor=model.video_encoder.vision_tower.processor,
        max_prompt_length=cfg["data"].get("max_prompt_tokens", 256),
        max_frames=cfg["data"]["max_frames"],
        action_dim=cfg["model"]["action_dim"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )

    for step, batch in enumerate(loader, start=1):
        metrics = trainer.train_step(
            LowLevelBatch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                video=batch["video"],
                proprio=batch["proprio"],
                target_actions=batch["target_actions"],
                fast_targets=batch["fast_targets"],
            )
        )
        print({"step": step, **metrics})
        if step >= cfg["trainer"].get("max_steps", 10):
            break


if __name__ == "__main__":
    main()
