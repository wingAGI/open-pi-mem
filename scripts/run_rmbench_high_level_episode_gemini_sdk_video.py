from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


def build_prompt(
    goal: str,
    prev_memory: str,
    planner_hz: float,
    step_index: int,
    clip_start_sec: float,
    clip_end_sec: float,
) -> str:
    return "".join([
        "You are the high-level planner for a robot policy.\n",
        "You are given a short video clip from an ongoing execution.\n",
        "The clip covers the most recent planning interval.\n",
        "The first planning call may contain only a near-instant initial observation because there is no earlier visual history yet.\n",
        f"You are called once every {planner_hz:.3f} Hz planning step.\n",
        f"This is inference step {step_index}.\n",
        f"This clip covers time {clip_start_sec:.2f}s to {clip_end_sec:.2f}s.\n",
        "Inputs:\n",
        "- Goal: the global task objective for the whole episode.\n",
        "- Previous memory: the state before this clip. It may be incomplete or wrong.\n",
        "- Current video clip: the most recent visual evidence over the latest planning interval.\n",
        "Outputs:\n",
        "- Memory: the state at the end of this clip.\n",
        "- Subtask: the single immediate next action to execute after this clip ends.\n",
        "Decision process:\n",
        "- First determine what changed during this clip.\n",
        "- Then update memory to reflect the end of this clip.\n",
        "- Then choose the next subtask.\n",
        "Constraints:\n",
        "- Use the video clip as the main evidence.\n",
        "- Reason about what action was being attempted, whether it likely succeeded, whether the stage is finished, and what state changed.\n",
        "- Subtask should be the very next executable action.\n",
        "- Keep subtask atomic and do not bundle multiple repetitions or multiple stages.\n",
        "- Memory should include only completed progress and current in-progress state at the end of this clip.\n",
        "- If an action has clearly finished within this clip, memory should describe it as completed.\n",
        "- If the action is still being executed or the completion is uncertain, memory should describe it as in progress.\n",
        "- Do not describe an in-progress action as completed.\n",
        "- Memory should not restate the full task requirements, future plan, or long explanations.\n",
        "Return exactly two lines and nothing else.\n",
        "Line 1 format: <subtask>...</subtask>\n",
        "Line 2 format: <memory>...</memory>\n",
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


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required binary '{name}' was not found in PATH.")
    return path


def _probe_float(video_path: Path, field: str) -> float:
    ffprobe = _require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        field,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    raw = result.stdout.strip()
    if not raw:
        raise RuntimeError(f"Could not read {field} from video: {video_path}")
    if "/" in raw:
        numerator, denominator = raw.split("/", maxsplit=1)
        return float(numerator) / max(float(denominator), 1.0)
    return float(raw)


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


def _probe_duration(video_path: Path) -> float:
    return _probe_float(video_path, "format=duration")


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


def _build_clip_windows(duration_sec: float, planner_hz: float, source_fps: float) -> list[tuple[float, float]]:
    total_steps = 1 + int(math.floor(duration_sec * planner_hz))
    initial_end = min(duration_sec, max(1.0 / max(source_fps, 1.0), 0.05))
    windows: list[tuple[float, float]] = [(0.0, initial_end)]
    for step_idx in range(1, total_steps):
        start = (step_idx - 1) / planner_hz
        end = min(duration_sec, step_idx / planner_hz)
        windows.append((start, end))
    return windows


def _extract_clip(video_path: Path, clip_path: Path, start_sec: float, end_sec: float) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    if clip_path.exists():
        return
    ffmpeg = _require_binary("ffmpeg")
    duration = max(end_sec - start_sec, 0.05)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-vf",
        "fps=10,scale=640:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        str(clip_path),
    ]
    subprocess.run(command, check=True)


def _extract_preview_frames(clip_path: Path, output_dir: Path, preview_fps: float) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.png"))
    if existing:
        return existing
    ffmpeg = _require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip_path),
        "-vf",
        f"fps={preview_fps:.6f}",
        str(output_dir / "frame_%03d.png"),
    ]
    subprocess.run(command, check=True)
    frames = sorted(output_dir.glob("frame_*.png"))
    if frames:
        return frames
    fallback = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip_path),
        "-frames:v",
        "1",
        str(output_dir / "frame_001.png"),
    ]
    subprocess.run(fallback, check=True)
    return sorted(output_dir.glob("frame_*.png"))


def _extract_response_payload(response: Any) -> tuple[str, str, int | None, bool]:
    texts: list[str] = []
    thoughts: list[str] = []
    thought_signature_present = False
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if getattr(part, "thought", None):
                if text:
                    thoughts.append(text)
            elif text:
                texts.append(text)
            if getattr(part, "thought_signature", None) or getattr(part, "thoughtSignature", None):
                thought_signature_present = True
    usage = getattr(response, "usage_metadata", None)
    thoughts_token_count = getattr(usage, "thoughts_token_count", None) if usage is not None else None
    return "\n".join(texts).strip(), "\n".join(thoughts).strip(), thoughts_token_count, thought_signature_present


def _run_single_clip(
    client: genai.Client,
    *,
    model_name: str,
    clip_path: Path,
    goal: str,
    prev_memory: str,
    planner_hz: float,
    step_index: int,
    clip_start_sec: float,
    clip_end_sec: float,
    max_output_tokens: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> tuple[str, str, str, str, dict[str, Any]]:
    prompt = build_prompt(
        goal,
        prev_memory,
        planner_hz,
        step_index,
        clip_start_sec,
        clip_end_sec,
    )
    contents: list[Any] = [
        prompt,
        types.Part.from_bytes(data=clip_path.read_bytes(), mime_type="video/mp4"),
    ]
    config = types.GenerateContentConfig(
        maxOutputTokens=max_output_tokens,
        temperature=0.2,
        thinkingConfig=types.ThinkingConfig(includeThoughts=True),
    )
    last_error: Exception | None = None
    for attempt in range(1, max(max_retries, 1) + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            text, thought_summary, thoughts_token_count, thought_signature_present = _extract_response_payload(response)
            next_subtask, next_memory = parse_prediction(text)
            metadata = {
                "thought_summary": thought_summary,
                "thoughts_token_count": thoughts_token_count,
                "thought_signature_present": thought_signature_present,
            }
            return prompt, text, next_subtask, next_memory, metadata
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
    source_fps: float,
    duration_sec: float,
    planner_hz: float,
    sampled_steps: int,
    model_name: str,
    records: list[dict[str, Any]],
) -> None:
    report_path.write_text(
        json.dumps(
            {
                "goal": goal,
                "source": source_label,
                "source_fps_assumed": source_fps,
                "source_duration_sec": duration_sec,
                "planner_hz": planner_hz,
                "sampled_frames": sampled_steps,
                "model_path": model_name,
                "provider": "gemini-sdk-video",
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_existing_report(report_path: Path) -> tuple[list[dict[str, Any]], str]:
    if not report_path.exists():
        return [], ""
    data = json.loads(report_path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not isinstance(records, list):
        records = []
    prev_memory = ""
    if records:
        last_memory = records[-1].get("next_memory")
        if isinstance(last_memory, str):
            prev_memory = last_memory
    return records, prev_memory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RMBench high-level episode inference with the Google Gemini SDK using short video clips."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--instruction-json", default=None)
    parser.add_argument("--hz", type=float, required=True, help="Planning frequency in Hz.")
    parser.add_argument("--prev-memory", default="")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-sec", type=float, default=3.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--preview-fps", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all planning steps.")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--update-memory", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing report.json in report-dir.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {args.api_key_env} is required.")
    if args.hz <= 0:
        raise ValueError("--hz must be > 0")

    video_path = Path(args.video).resolve()
    instruction_json = Path(args.instruction_json).resolve() if args.instruction_json else None
    goal = _load_goal(args.goal, instruction_json)
    client = genai.Client(api_key=api_key)

    def run_once(report_dir: Path | None) -> None:
        source_fps = _probe_fps(video_path)
        duration_sec = _probe_duration(video_path)
        windows = _build_clip_windows(duration_sec, args.hz, source_fps)
        source_label = str(video_path)
        if args.max_steps > 0:
            windows = windows[: args.max_steps]

        print(f"goal: {goal}")
        print(f"source: {source_label}")
        print(f"source_fps_assumed: {source_fps:.3f}")
        print(f"source_duration_sec: {duration_sec:.3f}")
        print(f"planner_hz: {args.hz:.3f}")
        print(f"sampled_steps: {len(windows)}")
        print(f"model: {args.model}")
        print()

        prev_memory = args.prev_memory
        records: list[dict[str, Any]] = []
        start_step = 0
        if args.resume and report_dir is not None:
            existing_records, resumed_memory = _load_existing_report(report_dir / "report.json")
            records = existing_records
            start_step = len(records)
            if resumed_memory:
                prev_memory = resumed_memory
            if start_step:
                print(f"resume_from_step: {start_step}")
                print(f"resume_prev_memory: {prev_memory}")
                print()

        for step_idx in range(start_step, len(windows)):
            clip_start_sec, clip_end_sec = windows[step_idx]
            if report_dir is not None:
                clip_path = report_dir / "clips" / f"step_{step_idx:03d}.mp4"
                preview_dir = report_dir / "preview_frames" / f"step_{step_idx:03d}"
            else:
                clip_path = Path(tempfile.gettempdir()) / f"gemini_video_step_{os.getpid()}_{step_idx:03d}.mp4"
                preview_dir = Path(tempfile.gettempdir()) / f"gemini_video_step_{os.getpid()}_{step_idx:03d}_frames"

            _extract_clip(video_path, clip_path, clip_start_sec, clip_end_sec)
            preview_frames = _extract_preview_frames(clip_path, preview_dir, args.preview_fps)

            input_memory = prev_memory
            prompt_text, raw_text, next_subtask, next_memory, gemini_meta = _run_single_clip(
                client,
                model_name=args.model,
                clip_path=clip_path,
                goal=goal,
                prev_memory=input_memory,
                planner_hz=args.hz,
                step_index=step_idx,
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                retry_backoff_sec=args.retry_backoff_sec,
            )
            timestamp_sec = round(step_idx / args.hz, 3)
            print(f"[step {step_idx:03d} | t={timestamp_sec:7.2f}s | clip={clip_start_sec:.2f}-{clip_end_sec:.2f}s]")
            print(f"prev_memory: {input_memory or '(none)'}")
            if gemini_meta.get("thought_summary"):
                print(f"thought_summary: {gemini_meta['thought_summary']}")
            print(f"next_subtask: {next_subtask or '[missing]'}")
            print(f"next_memory: {next_memory or '[missing]'}")
            print(f"raw: {raw_text or '[empty]'}")
            print()

            frame_times = []
            if preview_frames:
                if len(preview_frames) == 1:
                    frame_times = [round((clip_start_sec + clip_end_sec) / 2, 3)]
                else:
                    span = max(clip_end_sec - clip_start_sec, 1e-6)
                    frame_times = [
                        round(clip_start_sec + span * idx / (len(preview_frames) - 1), 3)
                        for idx in range(len(preview_frames))
                    ]

            records.append(
                {
                    "frame_index": step_idx,
                    "timestamp_sec": timestamp_sec,
                    "image_path": str(preview_frames[-1].relative_to(report_dir)) if report_dir is not None and preview_frames else "",
                    "input_image_paths": [
                        str(path.relative_to(report_dir)) if report_dir is not None else str(path)
                        for path in preview_frames
                    ],
                    "input_frame_timestamps_sec": frame_times,
                    "input_video_path": str(clip_path.relative_to(report_dir)) if report_dir is not None else str(clip_path),
                    "input_clip_start_sec": round(clip_start_sec, 3),
                    "input_clip_end_sec": round(clip_end_sec, 3),
                    "goal": goal,
                    "input_memory": input_memory,
                    "prompt": prompt_text,
                    "raw_output": raw_text,
                    "thought_summary": gemini_meta.get("thought_summary", ""),
                    "thoughts_token_count": gemini_meta.get("thoughts_token_count"),
                    "thought_signature_present": gemini_meta.get("thought_signature_present", False),
                    "next_subtask": next_subtask,
                    "next_memory": next_memory,
                }
            )

            if report_dir is not None:
                _write_report(
                    report_dir / "report.json",
                    goal=goal,
                    source_label=source_label,
                    source_fps=source_fps,
                    duration_sec=duration_sec,
                    planner_hz=args.hz,
                    sampled_steps=len(windows),
                    model_name=args.model,
                    records=records,
                )

            if args.update_memory and next_memory:
                prev_memory = next_memory

        if report_dir is not None:
            report_path = report_dir / "report.json"
            _write_report(
                report_path,
                goal=goal,
                source_label=source_label,
                source_fps=source_fps,
                duration_sec=duration_sec,
                planner_hz=args.hz,
                sampled_steps=len(windows),
                model_name=args.model,
                records=records,
            )
            print(f"report_json: {report_path}")

    if args.report_dir:
        report_dir = Path(args.report_dir).resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        run_once(report_dir)
    else:
        run_once(None)


if __name__ == "__main__":
    main()
