from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


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


def _encode_image_data_url(image_path: Path) -> str:
    mime_type = "image/png"
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _run_single_frame(
    client: OpenAI,
    *,
    model_name: str,
    api_mode: str,
    reasoning_effort: str | None,
    image_path: Path,
    goal: str,
    prev_memory: str,
    history_items: list[str],
    planner_hz: float,
    max_output_tokens: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> tuple[str, str, str, str]:
    prompt = build_prompt(goal, prev_memory, history_items, planner_hz)
    image_url = _encode_image_data_url(image_path)
    last_error: Exception | None = None
    for attempt in range(1, max(max_retries, 1) + 1):
        try:
            if api_mode == "chat":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    max_tokens=max_output_tokens,
                )
                text = (response.choices[0].message.content or "").strip()
            else:
                request_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": image_url},
                            ],
                        }
                    ],
                    "max_output_tokens": max_output_tokens,
                }
                if reasoning_effort and reasoning_effort != "none":
                    request_kwargs["reasoning"] = {"effort": reasoning_effort}
                response = client.responses.create(**request_kwargs)
                text = response.output_text.strip()
            next_subtask, next_memory = parse_prediction(text)
            return prompt, text, next_subtask, next_memory
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max(max_retries, 1):
                break
            time.sleep(max(retry_backoff_sec, 0.0) * attempt)
    assert last_error is not None
    raise last_error


def _write_report(
    report_path: Path,
    *,
    goal: str,
    source_label: str,
    fps: float,
    planner_hz: float,
    sampled_frames: int,
    model_name: str,
    base_url: str | None,
    api_mode: str,
    reasoning_effort: str,
    update_memory: bool,
    append_history: bool,
    records: list[dict[str, Any]],
) -> None:
    report_path.write_text(
        json.dumps(
            {
                "goal": goal,
                "source": source_label,
                "source_fps_assumed": fps,
                "planner_hz": planner_hz,
                "sampled_frames": sampled_frames,
                "model_path": model_name,
                "provider": "openai",
                "base_url": base_url,
                "api_mode": api_mode,
                "reasoning_effort": reasoning_effort,
                "update_memory": update_memory,
                "append_history": append_history,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _resolve_api_mode(api_mode: str, base_url: str | None) -> str:
    if api_mode in {"chat", "responses"}:
        return api_mode
    normalized = (base_url or "").lower()
    if "dashscope" in normalized or "bigmodel" in normalized:
        return "chat"
    return "responses"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RMBench high-level episode inference with OpenAI Responses API."
    )
    parser.add_argument("--video", required=True)
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
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-mode", default="auto", choices=["auto", "responses", "chat"])
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-sec", type=float, default=3.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all sampled frames.")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--update-memory", action="store_true")
    parser.add_argument("--append-history", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Environment variable {args.api_key_env} is required. "
            f"Create a key at https://platform.openai.com/api-keys and export it before running."
        )

    video_path = Path(args.video).resolve()
    instruction_json = Path(args.instruction_json).resolve() if args.instruction_json else None
    goal = _load_goal(args.goal, instruction_json)
    resolved_api_mode = _resolve_api_mode(args.api_mode, args.base_url)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    def run_once(frame_dir: Path, report_dir: Path | None) -> None:
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
        print(f"model: {args.model}")
        print(f"base_url: {args.base_url or 'default OpenAI'}")
        print(f"api_mode: {resolved_api_mode}")
        print()

        prev_memory = args.prev_memory
        history_items = list(args.history_item)
        records: list[dict[str, Any]] = []
        for idx, frame_path in enumerate(frames):
            input_memory = prev_memory
            prompt_text, raw_text, next_subtask, next_memory = _run_single_frame(
                client,
                model_name=args.model,
                api_mode=resolved_api_mode,
                reasoning_effort=args.reasoning_effort,
                image_path=frame_path,
                goal=goal,
                prev_memory=input_memory,
                history_items=history_items,
                planner_hz=args.hz,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                retry_backoff_sec=args.retry_backoff_sec,
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

            if report_dir is not None:
                _write_report(
                    report_dir / "report.json",
                    goal=goal,
                    source_label=source_label,
                    fps=fps,
                    planner_hz=args.hz,
                    sampled_frames=len(frames),
                    model_name=args.model,
                    base_url=args.base_url,
                    api_mode=resolved_api_mode,
                    reasoning_effort=args.reasoning_effort,
                    update_memory=bool(args.update_memory),
                    append_history=bool(args.append_history),
                    records=records,
                )

            if args.update_memory and next_memory:
                prev_memory = next_memory
            if args.append_history and next_subtask:
                history_items.append(f"{next_subtask} | status=success")

        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "report.json"
            _write_report(
                report_path,
                goal=goal,
                source_label=source_label,
                fps=fps,
                planner_hz=args.hz,
                sampled_frames=len(frames),
                model_name=args.model,
                base_url=args.base_url,
                api_mode=resolved_api_mode,
                reasoning_effort=args.reasoning_effort,
                update_memory=bool(args.update_memory),
                append_history=bool(args.append_history),
                records=records,
            )
            print(f"report_json: {report_path}")

    if args.report_dir:
        report_dir = Path(args.report_dir).resolve()
        frame_dir = report_dir / "frames"
        run_once(frame_dir, report_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="rmbench_hl_openai_") as tmp_dir:
            run_once(Path(tmp_dir), None)


if __name__ == "__main__":
    main()
