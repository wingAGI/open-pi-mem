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


def _encode_image_data_url(image_path: Path) -> str:
    mime_type = "image/png"
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _build_windows(sampled_frames: list[Path], window_frames: int) -> list[list[Path]]:
    if not sampled_frames:
        return []
    windows: list[list[Path]] = [[sampled_frames[0]]]
    start = 1
    while start < len(sampled_frames):
        windows.append(sampled_frames[start : start + window_frames])
        start += window_frames
    return windows


def _run_single_window(
    client: OpenAI,
    *,
    model_name: str,
    api_mode: str,
    reasoning_effort: str | None,
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
) -> tuple[str, str, str, str]:
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
    image_urls = [_encode_image_data_url(path) for path in image_paths]
    last_error: Exception | None = None
    for attempt in range(1, max(max_retries, 1) + 1):
        try:
            if api_mode == "chat":
                content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                for image_url in image_urls:
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_output_tokens,
                )
                text = (response.choices[0].message.content or "").strip()
            else:
                content = [{"type": "input_text", "text": prompt}]
                for image_url in image_urls:
                    content.append({"type": "input_image", "image_url": image_url})
                request_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "input": [{"role": "user", "content": content}],
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
    source_fps: float,
    planner_hz: float,
    dense_sampling_hz: float,
    window_frames: int,
    sampled_steps: int,
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
                "source_fps_assumed": source_fps,
                "planner_hz": planner_hz,
                "dense_sampling_hz": dense_sampling_hz,
                "window_frames": window_frames,
                "sampled_frames": sampled_steps,
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
        description="Run RMBench high-level episode inference with an OpenAI-compatible API using a multi-frame window."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--instruction-json", default=None)
    parser.add_argument("--hz", type=float, required=True, help="Planning frequency in Hz.")
    parser.add_argument("--window-frames", type=int, default=3, help="Number of recent frames per planning step.")
    parser.add_argument("--prev-memory", default="")
    parser.add_argument(
        "--history-item",
        action="append",
        default=[],
        help='Repeatable history item, e.g. --history-item "reach handle | status=success"',
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-mode", default="auto", choices=["auto", "responses", "chat"])
    parser.add_argument("--reasoning-effort", default="medium", choices=["none", "low", "medium", "high", "xhigh"])
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-sec", type=float, default=3.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all planning steps.")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--update-memory", action="store_true")
    parser.add_argument("--append-history", action="store_true")
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
    resolved_api_mode = _resolve_api_mode(args.api_mode, args.base_url)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

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
        print(f"base_url: {args.base_url or 'default OpenAI'}")
        print(f"api_mode: {resolved_api_mode}")
        print()

        prev_memory = args.prev_memory
        records: list[dict[str, Any]] = []
        for step_idx, image_paths in enumerate(windows):
            input_memory = prev_memory
            frame_timestamps_sec = [
                round((step_idx * args.window_frames + local_idx) / dense_sampling_hz, 3)
                for local_idx in range(len(image_paths))
            ]
            prompt_text, raw_text, next_subtask, next_memory = _run_single_window(
                client,
                model_name=args.model,
                api_mode=resolved_api_mode,
                reasoning_effort=args.reasoning_effort,
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
                    base_url=args.base_url,
                    api_mode=resolved_api_mode,
                    reasoning_effort=args.reasoning_effort,
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
        with tempfile.TemporaryDirectory(prefix="rmbench_hl_openai_multiframe_") as tmp_dir:
            run_once(Path(tmp_dir), None)


if __name__ == "__main__":
    main()
