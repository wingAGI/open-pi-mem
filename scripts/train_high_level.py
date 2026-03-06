from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from open_pi_mem.data.training_datasets import HighLevelCollator, MemorySupervisionDataset
from open_pi_mem.models.high_level_policy import HighLevelPolicy
from open_pi_mem.training.common import set_seed
from open_pi_mem.training.high_level_trainer import HighLevelBatch, HighLevelTrainer
from open_pi_mem.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 7))

    model = HighLevelPolicy(cfg["model"])
    trainer = HighLevelTrainer(
        model,
        learning_rate=cfg["trainer"]["learning_rate"],
        weight_decay=cfg["trainer"].get("weight_decay", 0.01),
    )
    dataset = MemorySupervisionDataset(cfg["data"]["train_jsonl"])
    collator = HighLevelCollator(
        tokenizer=model.text_backbone.tokenizer,
        max_length=cfg["data"].get("max_total_tokens", 512),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )

    for step, batch in enumerate(loader, start=1):
        metrics = trainer.train_step(
            HighLevelBatch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        )
        print({"step": step, **metrics})
        if step >= cfg["trainer"].get("max_steps", 10):
            break


if __name__ == "__main__":
    main()
