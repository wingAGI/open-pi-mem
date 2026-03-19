from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image

from open_pi_mem.models.high_level_policy import HighLevelPolicy
from open_pi_mem.utils.config import load_yaml


def build_prompt(goal: str, prev_memory: str, history_items: list[str], planner_hz: float | None = None) -> str:
    history_text = "\n".join(history_items) if history_items else "- None | status=unknown"
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
        "- History: optional textual history of earlier subtasks and outcomes. It is auxiliary context only.\n",
        "Meaning of the outputs:\n",
        "- Subtask: the single immediate next action that should be executed now by the low-level policy.\n",
        "- Memory: a short natural-language summary of what has ALREADY been completed or is ALREADY true at the current moment, after considering the current image.\n",
        "Use the CURRENT IMAGE as the primary evidence for progress.\n",
        "Previous memory and history are only weak hints and may be wrong.\n",
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
        "Good memory example: <memory>Left button pressed once.</memory>\n",
        "Bad memory example: <memory>Press left button once more, then press middle button seven times.</memory>\n",
        "Good subtask example: <subtask>Press the left button once.</subtask>\n",
        "Bad subtask example: <subtask>Press the left button twice.</subtask>\n",
        "Bad subtask example: <subtask>Press left twice, then middle seven times, then confirm.</subtask>\n",
        "Bad subtask example: <subtask>Press the left button once, then press the middle button.</subtask>\n",
        "Do not include rationale, bullets, prefixes, or extra commentary.\n",
        f"Goal: {goal}\n",
        f"Previous memory: {prev_memory or 'None'}\n",
        f"History:\n{history_text}\n",
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


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required binary '{name}' was not found in PATH.")
    return path


def _probe_fps(video_path: Path) -> float:
    ffprobe = _require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    raw = result.stdout.strip()
    if not raw:
        raise RuntimeError(f"Could not read FPS from video: {video_path}")
    if "/" in raw:
        numerator, denominator = raw.split("/", maxsplit=1)
        return float(numerator) / max(float(denominator), 1.0)
    return float(raw)


def _extract_frames(video_path: Path, hz: float, output_dir: Path) -> list[Path]:
    if hz <= 0:
        raise ValueError("--hz must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = _probe_fps(video_path)
    frame_interval = max(int(round(fps / hz)), 1)
    ffmpeg = _require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select='not(mod(n\\,{frame_interval}))'",
        "-vsync",
        "vfr",
        str(output_dir / "frame_%06d.png"),
    ]
    subprocess.run(command, check=True)
    return sorted(output_dir.glob("frame_*.png"))


def _find_hdf5_image_dataset(h5_file: "h5py.File") -> str:
    image_candidates: list[tuple[int, str, tuple[int, ...]]] = []
    byte_candidates: list[tuple[int, str, tuple[int, ...]]] = []

    def visitor(name: str, obj: object) -> None:
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)
        if shape is None or dtype is None:
            return
        shape_tuple = tuple(int(v) for v in shape)
        if len(shape_tuple) == 4 and shape_tuple[-1] == 3:
            priority = 0 if "head_camera" in name else 1
            image_candidates.append((priority, name, shape_tuple))
            return
        if len(shape_tuple) == 1 and dtype.kind in {"S", "O"}:
            priority = 0 if "head_camera" in name else 1 if "third_view" in name else 2
            byte_candidates.append((priority, name, shape_tuple))

    h5_file.visititems(visitor)
    if image_candidates:
        image_candidates.sort(key=lambda item: (item[0], -item[2][0]))
        return image_candidates[0][1]
    if byte_candidates:
        byte_candidates.sort(key=lambda item: (item[0], -item[2][0]))
        return byte_candidates[0][1]
    raise RuntimeError("Could not find a supported image dataset in the HDF5 file.")


def _decode_hdf5_frame(item: object) -> Image.Image:
    if isinstance(item, (bytes, bytearray)):
        return Image.open(io.BytesIO(item)).convert("RGB")
    if hasattr(item, "tobytes") and not hasattr(item, "shape"):
        return Image.open(io.BytesIO(item.tobytes())).convert("RGB")
    array = np.asarray(item)
    if array.dtype.kind in {"S", "O"}:
        return Image.open(io.BytesIO(bytes(array.tolist()))).convert("RGB")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _extract_frames_from_hdf5(hdf5_path: Path, hz: float, output_dir: Path) -> list[Path]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading HDF5 episodes requires h5py to be installed.") from exc

    if hz <= 0:
        raise ValueError("--hz must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        dataset_name = _find_hdf5_image_dataset(f)
        dataset = f[dataset_name]
        total_frames = int(dataset.shape[0])
        if total_frames == 0:
            raise RuntimeError(f"No frames found in dataset '{dataset_name}'")
        step = max(int(round(30.0 / hz)), 1)
        sampled_indices = list(range(0, total_frames, step))
        frame_paths: list[Path] = []
        for out_idx, frame_idx in enumerate(sampled_indices):
            path = output_dir / f"frame_{out_idx:06d}.png"
            image = _decode_hdf5_frame(dataset[frame_idx])
            image.save(path)
            frame_paths.append(path)
    return frame_paths


def _load_goal(goal: str | None, instruction_json: Path | None) -> str:
    if goal:
        return goal
    if instruction_json is None:
        raise ValueError("Provide either --goal or --instruction-json.")
    with instruction_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("unseen", "seen"):
        values = data.get(key)
        if isinstance(values, list) and values:
            return str(values[0])
    raise ValueError(f"Could not find goal text in {instruction_json}")


def _build_model(cfg: dict, device: torch.device) -> HighLevelPolicy:
    model = HighLevelPolicy(cfg["model"])
    model.to(device)
    model.eval()
    return model


def _run_single_frame(
    model: HighLevelPolicy,
    cfg: dict,
    device: torch.device,
    image_path: Path,
    goal: str,
    prev_memory: str,
    history_items: list[str],
    max_new_tokens: int,
) -> tuple[str, str, str]:
    prompt = build_prompt(goal, prev_memory, history_items, cfg.get("planner_hz"))
    image = Image.open(image_path).convert("RGB")

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
            max_new_tokens=max_new_tokens,
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
    return prompt, text.strip(), next_subtask, next_memory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample an RMBench episode video at a configurable high-level planning rate and run high-level inference."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--hdf5", default=None)
    parser.add_argument("--model-path", default=None, help="Override model.multimodal_backbone_name with a local or remote model path.")
    parser.add_argument("--goal", default=None)
    parser.add_argument("--instruction-json", default=None)
    parser.add_argument("--hz", type=float, required=True, help="High-level planning frequency in Hz.")
    parser.add_argument("--prev-memory", default="")
    parser.add_argument(
        "--history-item",
        action="append",
        default=[],
        help='Repeatable history item, e.g. --history-item "reach handle | status=success"',
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all sampled frames.")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Optional directory to save sampled frames and a JSON report for visualization.",
    )
    parser.add_argument(
        "--update-memory",
        action="store_true",
        help="After each step, feed the predicted memory into the next step.",
    )
    parser.add_argument(
        "--append-history",
        action="store_true",
        help="After each step, append the predicted subtask as a success item in history for the next step.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["planner_hz"] = args.hz
    if args.model_path:
        cfg.setdefault("model", {})
        cfg["model"]["multimodal_backbone_name"] = args.model_path
        cfg["model"]["local_files_only"] = True
    if not args.video and not args.hdf5:
        raise ValueError("Provide either --video or --hdf5.")
    video_path = Path(args.video).resolve() if args.video else None
    hdf5_path = Path(args.hdf5).resolve() if args.hdf5 else None
    instruction_json = Path(args.instruction_json).resolve() if args.instruction_json else None
    goal = _load_goal(args.goal, instruction_json)

    device_name = cfg.get("trainer", {}).get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = _build_model(cfg, device)

    def run_once(frame_dir: Path, report_dir: Path | None) -> None:
        if hdf5_path is not None:
            frames = _extract_frames_from_hdf5(hdf5_path, args.hz, frame_dir)
            fps = 30.0
            source_label = str(hdf5_path)
        else:
            assert video_path is not None
            frames = _extract_frames(video_path, args.hz, frame_dir)
            fps = _probe_fps(video_path)
            source_label = str(video_path)
        if not frames:
            raise RuntimeError(f"No frames were extracted from {source_label}")
        if args.max_frames > 0:
            frames = frames[: args.max_frames]

        sampled_interval_sec = max(1.0 / fps, 0.0) * max(int(round(fps / args.hz)), 1)
        print(f"goal: {goal}")
        print(f"source: {source_label}")
        print(f"source_fps_assumed: {fps:.3f}")
        print(f"planner_hz: {args.hz:.3f}")
        print(f"sampled_frames: {len(frames)}")
        print()

        prev_memory = args.prev_memory
        history_items = list(args.history_item)
        records: list[dict[str, Any]] = []
        for idx, frame_path in enumerate(frames):
            input_memory = prev_memory
            prompt_text, raw_text, next_subtask, next_memory = _run_single_frame(
                model=model,
                cfg=cfg,
                device=device,
                image_path=frame_path,
                goal=goal,
                prev_memory=input_memory,
                history_items=history_items,
                max_new_tokens=args.max_new_tokens,
            )
            timestamp_sec = idx * sampled_interval_sec
            print(f"[frame {idx:03d} | t={timestamp_sec:7.2f}s | image={frame_path.name}]")
            print(f"prev_memory: {input_memory or '(none)'}")
            print(f"next_subtask: {next_subtask or '[missing]'}")
            print(f"next_memory: {next_memory or '[missing]'}")
            print(f"raw: {raw_text or '[empty]'}")
            print()

            records.append(
                {
                    "frame_index": idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "image_path": str(frame_path.relative_to(report_dir)) if report_dir is not None else str(frame_path),
                    "goal": goal,
                    "input_memory": input_memory,
                    "history_items": list(history_items),
                    "prompt": prompt_text,
                    "raw_output": raw_text,
                    "next_subtask": next_subtask,
                    "next_memory": next_memory,
                }
            )

            if args.update_memory and next_memory:
                prev_memory = next_memory
            if args.append_history and next_subtask:
                history_items.append(f"{next_subtask} | status=success")

        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "goal": goal,
                        "source": source_label,
                        "source_fps_assumed": fps,
                        "planner_hz": args.hz,
                        "sampled_frames": len(frames),
                        "model_path": cfg["model"].get("multimodal_backbone_name"),
                        "update_memory": bool(args.update_memory),
                        "append_history": bool(args.append_history),
                        "records": records,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"report_json: {report_path}")

    if args.report_dir:
        report_dir = Path(args.report_dir).resolve()
        frame_dir = report_dir / "frames"
        run_once(frame_dir, report_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="rmbench_hl_") as tmp_dir:
            run_once(Path(tmp_dir), None)


if __name__ == "__main__":
    main()
