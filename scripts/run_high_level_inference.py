from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image

from open_pi_mem.models.high_level_policy import HighLevelPolicy
from open_pi_mem.utils.config import load_yaml


def load_checkpoint(model: HighLevelPolicy, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }
    model.load_state_dict(checkpoint, strict=False)


def build_prompt(goal: str, prev_memory: str, history_items: list[str], planner_hz: float | None = None) -> str:
    cadence_text = (
        f"You are called once every {planner_hz:.3f} Hz planning step. After the current subtask is executed for one planning interval, you will receive a new image.\n"
        if planner_hz is not None
        else "You are called at a fixed planning rate. After the current subtask is executed for one planning interval, you will receive a new image.\n"
    )
    return "".join([
        "You are the high-level planner for a robot policy.\n",
        "You are seeing ONE frame from an ongoing execution, not a summary of the whole episode.\n",
        "The workflow is:\n",
        "1. At the beginning, you receive the global goal, the initial image, and an empty previous memory.\n",
        "2. You output one next subtask and one next memory.\n",
        "3. The next subtask is sent to the low-level policy and executed.\n",
        "4. Your next memory is fed back to you as previous memory at the next planning step.\n",
        "5. After one planning interval, you receive a new image and must plan again.\n",
        cadence_text,
        "Meaning of the inputs:\n",
        "- Goal: the global task objective for the whole episode. It usually stays the same across the whole rollout.\n",
        "- Previous memory: your own memory from the PREVIOUS planning step. It summarizes what was already completed or already true before the current decision. It may be incomplete or wrong.\n",
        "Meaning of the outputs:\n",
        "- Subtask: the single immediate next action that should be executed now by the low-level policy.\n",
        "- Memory: a short natural-language summary of what has ALREADY been completed or is ALREADY true at the current moment, after considering the current image.\n",
        "Use the CURRENT IMAGE as the primary evidence for progress.\n",
        "Do NOT assume a subtask is finished unless the current image provides clear visual evidence.\n",
        "If the image is ambiguous, be conservative: keep the current subtask or keep the state unfinished.\n",
        "Do not advance just because advancing sounds logically plausible from the goal.\n",
        "Only update memory with information supported by the current image and the goal.\n",
        "Memory must describe ONLY what is already completed or already confirmed true.\n",
        "Memory must NOT contain future plans, instructions, rationale, or what to do next.\n",
        "Memory should be short, factual, and past-tense when possible.\n",
        "Subtask must describe ONLY the immediate next action to execute now.\n",
        "Subtask must be ATOMIC: it should describe one small action that cannot be usefully split into smaller repeated actions at this planning frequency.\n",
        "If a step can be decomposed into repeated single actions, output only ONE repetition as the subtask.\n",
        "For example, prefer 'Press the left button once.' over 'Press the left button twice.'\n",
        "Do not bundle multiple repetitions or multiple stages into one subtask.\n",
        "Subtask must NOT summarize the whole plan.\n",
        "If the task already appears completed in the current image, say so briefly in the memory and output <subtask>None</subtask>.\n",
        "Time-consistency rule:\n",
        "- Previous memory is the state BEFORE this decision.\n",
        "- Next memory is the state you want to carry into the NEXT decision.\n",
        "- Therefore, next memory and next subtask must be temporally consistent: next memory should not claim that the next subtask has already been completed unless the current image clearly shows it is already done.\n",
        "Return exactly two lines and nothing else.\n",
        "Line 1 format: <subtask>...</subtask>\n",
        "Line 2 format: <memory>...</memory>\n",
        "Good subtask example: <subtask>Press the left button once.</subtask>\n",
        "Bad subtask example: <subtask>Press the left button twice.</subtask>\n",
        "Bad subtask example: <subtask>Press the left button once, then press the middle button.</subtask>\n",
        f"Goal: {goal}\n",
        f"Previous memory: {prev_memory or 'None'}\n",
        "Predict the next subtask and next memory.\n",
    ])


def parse_prediction(text: str) -> tuple[str, str]:
    subtask_match = re.search(r"<subtask>(.*?)</subtask>", text, flags=re.DOTALL | re.IGNORECASE)
    memory_match = re.search(r"<memory>(.*?)</memory>", text, flags=re.DOTALL | re.IGNORECASE)
    if subtask_match:
        subtask = subtask_match.group(1).strip()
    else:
        alt_subtask = re.search(
            r"^\s*(?:next\s+subtask|subtask)\s*:\s*(.+)$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        subtask = alt_subtask.group(1).strip() if alt_subtask else ""
    if memory_match:
        memory = memory_match.group(1).strip()
    else:
        alt_memory = re.search(
            r"^\s*(?:next\s+memory|memory\s+update|memory)\s*:\s*(.+)$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        memory = alt_memory.group(1).strip() if alt_memory else ""
    return subtask, memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run high-level MEM inference on one image and text state.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--prev-memory", default="")
    parser.add_argument("--checkpoint", default=None, help="Optional fine-tuned checkpoint to load.")
    parser.add_argument("--model-path", default=None, help="Optional local or remote VLM path to override config.")
    parser.add_argument("--local-files-only", action="store_true", help="Force transformers to load only local files.")
    parser.add_argument("--planner-hz", type=float, default=None, help="Optional planning frequency for the prompt.")
    parser.add_argument(
        "--history-item",
        action="append",
        default=[],
        help='Repeatable history item, e.g. --history-item "reach handle | status=success"',
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("model", {})
    if args.model_path:
        cfg["model"]["multimodal_backbone_name"] = args.model_path
    if args.local_files_only:
        cfg["model"]["local_files_only"] = True
    device_name = cfg.get("trainer", {}).get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = HighLevelPolicy(cfg["model"])
    model.to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    model.eval()

    prompt = build_prompt(args.goal, args.prev_memory, args.history_item, planner_hz=args.planner_hz)
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
