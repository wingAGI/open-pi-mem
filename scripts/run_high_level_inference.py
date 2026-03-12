from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image

from open_pi_mem.models.high_level_policy import HighLevelPolicy
from open_pi_mem.utils.config import load_yaml


def build_prompt(goal: str, prev_memory: str, history_items: list[str]) -> str:
    history_text = render_history(history_items)
    return (
        "You are the high-level planner for a robot policy.\n"
        "Return exactly two XML fields and nothing else.\n"
        "<subtask>...</subtask>\n"
        "<memory>...</memory>\n"
        "Rules:\n"
        "- Output only these two lines.\n"
        "- Do not add explanations.\n"
        "- The subtask should be the next concrete action-level instruction.\n"
        "- The memory should be a short planning state summary for future steps.\n"
        "- If the previous memory is still valid, keep it concise and update only what changed.\n"
        "- The history is provided as XML events; use status to avoid repeating completed steps.\n"
        "Example:\n"
        "<subtask>pull fridge door outward</subtask>\n"
        "<memory>reached the fridge handle; fridge door partly open</memory>\n"
        f"Goal: {goal}\n"
        f"Previous memory: {prev_memory or 'None'}\n"
        f"History:\n{history_text}\n"
        "Predict the next subtask and memory update.\n"
    )


def render_history(history_items: list[str]) -> str:
    if not history_items:
        return '<event><subtask>None</subtask><status>unknown</status></event>'
    return "\n".join(_history_item_to_xml(item) for item in history_items)


def _history_item_to_xml(item: str) -> str:
    text = item.strip()
    if "|" not in text:
        return f"<event><subtask>{text}</subtask><status>unknown</status></event>"
    subtask_part, meta_part = text.split("|", 1)
    subtask = subtask_part.strip()
    status = "unknown"
    for field in meta_part.split("|"):
        field = field.strip()
        if field.startswith("status="):
            status = field.split("=", 1)[1].strip() or "unknown"
            break
    return f"<event><subtask>{subtask}</subtask><status>{status}</status></event>"


def parse_prediction(text: str) -> tuple[str, str]:
    subtask_match = re.search(r"<subtask>(.*?)</subtask>", text, flags=re.DOTALL)
    memory_match = re.search(r"<memory>(.*?)</memory>", text, flags=re.DOTALL)
    subtask = subtask_match.group(1).strip() if subtask_match else ""
    memory = memory_match.group(1).strip() if memory_match else ""
    if subtask or memory:
        return subtask, memory
    cleaned = text.strip()
    if not cleaned:
        return "", ""
    first_line = cleaned.splitlines()[0].strip()
    if first_line:
        return first_line, prev_like_memory(cleaned)
    return "", ""


def prev_like_memory(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return ""
    return " ".join(lines[1:]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run high-level MEM inference on one image and text state.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--prev-memory", default="")
    parser.add_argument(
        "--history-item",
        action="append",
        default=[],
        help='Repeatable history item, e.g. --history-item "reach handle | status=success"',
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device_name = cfg.get("trainer", {}).get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = HighLevelPolicy(cfg["model"])
    model.to(device)
    model.eval()

    prompt = build_prompt(args.goal, args.prev_memory, args.history_item)
    image = Image.open(args.image).convert("RGB")

    if model.is_multimodal:
        prompt_text = model.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = model.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )
    else:
        batch = model.text_backbone.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["data"].get("max_total_tokens", 512),
        )

    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        generated = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_tokens = generated[:, batch["input_ids"].shape[1] :]
    if model.is_multimodal:
        text = model.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    else:
        text = model.text_backbone.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    next_subtask, next_memory = parse_prediction(text)
    print("Raw output:")
    print(text.strip())
    print("\nParsed prediction:")
    print(f"next_subtask: {next_subtask or '[missing]'}")
    print(f"next_memory: {next_memory or '[missing]'}")


if __name__ == "__main__":
    main()
