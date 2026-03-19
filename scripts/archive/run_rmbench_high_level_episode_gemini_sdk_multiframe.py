from __future__ import annotations

import argparse
import json
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
    window_frames: int,
    step_index: int,
    window_start_sec: float,
    window_end_sec: float,
    frame_timestamps_text: str,
) -> str:
    return "".join([
        "You are the high-level planner for a robot policy.\n",
        "You are given a short ordered sequence of consecutive frames from an ongoing execution.\n",
        "The frames are arranged from oldest to newest.\n",
        "They describe the most recent planning interval.\n",
        "The first planning call may contain only one frame because there is no earlier visual history yet.\n",
        f"You are called once every {planner_hz:.3f} Hz planning step.\n",
        f"Each planning step receives up to {window_frames} recent frames covering the latest planning interval.\n",
        f"This is inference step {step_index}.\n",
        f"This frame window covers time {window_start_sec:.2f}s to {window_end_sec:.2f}s.\n",
        f"Frame timestamps:\n{frame_timestamps_text}\n",
        "Inputs:\n",
        "- Goal: the global task objective for the whole episode.\n",
        "- Previous memory: the state before this frame window. It may be incomplete or wrong.\n",
        "- Current frame sequence: the recent observations in time order from oldest to newest.\n",
        "Outputs:\n",
        "- Memory: the state at the end of this frame window.\n",
        "- Subtask: the single immediate next action to execute after this frame window ends.\n",
        "Decision process:\n",
        "- First determine what changed across this frame window.\n",
        "- Then update memory to reflect the end of this frame window.\n",
        "- Then choose the next subtask.\n",
        "Constraints:\n",
        "- Use the frame sequence as the main evidence.\n",
        "- Reason about what action was being attempted, whether it likely succeeded, whether the stage is finished, and what state changed.\n",
        "- Subtask: the single immediate next action to execute after the current frame sequence ends.\n",
        "- Subtask should be the very next executable action.\n",
        "- Keep subtask atomic and do not bundle multiple repetitions or multiple stages.\n",
        "- Memory should include only completed progress and current in-progress state at the end of this frame window.\n",
        "- If an action has clearly finished within this frame window, memory should describe it as completed.\n",
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


def _extract_dense_frames(video_path: Path, sampling_hz: float, output_dir: Path) -> list[Path]:
    if sampling_hz <= 0:
        raise ValueError("sampling_hz must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_dir.glob("frame_*.png"))
    if existing_frames:
        return existing_frames
    ffmpeg = _require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sampling_hz:.6f}",
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


def _build_windows(sampled_frames: list[Path], window_frames: int) -> list[list[Path]]:
    if not sampled_frames:
        return []
    windows: list[list[Path]] = [[sampled_frames[0]]]
    start = 1
    while start < len(sampled_frames):
        windows.append(sampled_frames[start : start + window_frames])
        start += window_frames
    return windows


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


def _run_single_window(
    client: genai.Client,
    *,
    model_name: str,
    image_paths: list[Path],
    goal: str,
    prev_memory: str,
    planner_hz: float,
    window_frames: int,
    step_index: int,
    frame_timestamps_sec: list[float],
    max_output_tokens: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> tuple[str, str, str, str, dict[str, Any]]:
    frame_timestamps_text = "\n".join(
        f"- frame {idx + 1}: t={timestamp:.2f}s"
        for idx, timestamp in enumerate(frame_timestamps_sec)
    )
    prompt = build_prompt(
        goal,
        prev_memory,
        planner_hz,
        window_frames,
        step_index,
        frame_timestamps_sec[0],
        frame_timestamps_sec[-1],
        frame_timestamps_text,
    )

    contents: list[Any] = [prompt] + [types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png") for path in image_paths]

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
    planner_hz: float,
    dense_sampling_hz: float,
    window_frames: int,
    sampled_steps: int,
    model_name: str,
    update_memory: bool,
    append_history: bool,
    records: list[dict[str, Any]],
) -> None:
    report_path.write_text(
        json.dumps(
            {
                "goal": goal,
                "source": source_label,
                "source_fps_assumed": source_fps,
                "planner_hz": planner_hz,
                "dense_sampling_hz": dense_sampling_hz,
                "window_frames": window_frames,
                "sampled_frames": sampled_steps,
                "model_path": model_name,
                "provider": "gemini-sdk",
                "update_memory": update_memory,
                "append_history": append_history,
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
        description="Run RMBench high-level episode inference with the Google Gemini SDK using a multi-frame window."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--instruction-json", default=None)
    parser.add_argument("--hz", type=float, required=True, help="Planning frequency in Hz.")
    parser.add_argument("--window-frames", type=int, default=3, help="Number of recent frames per planning step.")
    parser.add_argument("--prev-memory", default="")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-sec", type=float, default=3.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all planning steps.")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--update-memory", action="store_true")
    parser.add_argument("--append-history", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing report.json in report-dir.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {args.api_key_env} is required.")
    if args.hz <= 0:
        raise ValueError("--hz must be > 0")
    if args.window_frames <= 0:
        raise ValueError("--window-frames must be > 0")

    video_path = Path(args.video).resolve()
    instruction_json = Path(args.instruction_json).resolve() if args.instruction_json else None
    goal = _load_goal(args.goal, instruction_json)
    client = genai.Client(api_key=api_key)

    def run_once(frame_dir: Path, report_dir: Path | None) -> None:
        dense_sampling_hz = args.hz * args.window_frames
        dense_frames = _extract_dense_frames(video_path, dense_sampling_hz, frame_dir)
        windows = _build_windows(dense_frames, args.window_frames)
        source_fps = _probe_fps(video_path)
        source_label = str(video_path)
        if not windows:
            raise RuntimeError(f"No frames were extracted from {source_label}")
        if args.max_steps > 0:
            windows = windows[: args.max_steps]

        print(f"goal: {goal}")
        print(f"source: {source_label}")
        print(f"source_fps_assumed: {source_fps:.3f}")
        print(f"planner_hz: {args.hz:.3f}")
        print(f"dense_sampling_hz: {dense_sampling_hz:.3f}")
        print(f"window_frames: {args.window_frames}")
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
            image_paths = windows[step_idx]
            input_memory = prev_memory
            frame_timestamps_sec = [
                round((step_idx * args.window_frames + local_idx) / dense_sampling_hz, 3)
                for local_idx in range(len(image_paths))
            ]
            prompt_text, raw_text, next_subtask, next_memory, gemini_meta = _run_single_window(
                client,
                model_name=args.model,
                image_paths=image_paths,
                goal=goal,
                prev_memory=input_memory,
                planner_hz=args.hz,
                window_frames=args.window_frames,
                step_index=step_idx,
                frame_timestamps_sec=frame_timestamps_sec,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                retry_backoff_sec=args.retry_backoff_sec,
            )
            timestamp_sec = round(step_idx / args.hz, 3)
            print(
                f"[step {step_idx:03d} | t={timestamp_sec:7.2f}s | frames="
                + ", ".join(path.name for path in image_paths)
                + "]"
            )
            print(f"prev_memory: {input_memory or '(none)'}")
            if gemini_meta.get("thought_summary"):
                print(f"thought_summary: {gemini_meta['thought_summary']}")
            print(f"next_subtask: {next_subtask or '[missing]'}")
            print(f"next_memory: {next_memory or '[missing]'}")
            print(f"raw: {raw_text or '[empty]'}")
            print()

            records.append(
                {
                    "frame_index": step_idx,
                    "timestamp_sec": timestamp_sec,
                    "image_path": str(image_paths[-1].relative_to(report_dir)) if report_dir is not None else str(image_paths[-1]),
                    "input_image_paths": [
                        str(path.relative_to(report_dir)) if report_dir is not None else str(path)
                        for path in image_paths
                    ],
                    "input_frame_timestamps_sec": frame_timestamps_sec,
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
                    planner_hz=args.hz,
                    dense_sampling_hz=dense_sampling_hz,
                    window_frames=args.window_frames,
                    sampled_steps=len(windows),
                    model_name=args.model,
                    update_memory=bool(args.update_memory),
                    append_history=bool(args.append_history),
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
                planner_hz=args.hz,
                dense_sampling_hz=dense_sampling_hz,
                window_frames=args.window_frames,
                sampled_steps=len(windows),
                model_name=args.model,
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
        with tempfile.TemporaryDirectory(prefix="rmbench_hl_gemini_sdk_multiframe_") as tmp_dir:
            run_once(Path(tmp_dir), None)


if __name__ == "__main__":
    main()
