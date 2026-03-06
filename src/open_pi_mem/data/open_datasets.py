from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from open_pi_mem.data.schemas import LowLevelTrainingRecord
from open_pi_mem.utils.io import write_jsonl


@dataclass(frozen=True)
class DatasetCandidate:
    name: str
    role: str
    note: str


OPEN_DATASET_CANDIDATES = [
    DatasetCandidate("open_x_embodiment", "robot_demos", "broad cross-robot imitation data"),
    DatasetCandidate("bridge_v2", "robot_demos", "language-conditioned tabletop manipulation via RLDS exports"),
    DatasetCandidate("droid", "robot_demos", "teleoperated household manipulation via RLDS exports"),
    DatasetCandidate("robomimic", "robot_demos", "benchmark-scale imitation learning"),
    DatasetCandidate("ego4d", "long_video", "egocentric event structure and narration"),
    DatasetCandidate("epic_kitchens", "long_video", "kitchen actions and temporal segmentation"),
]


class RLDSWindowBuilder:
    """Convert local RLDS/TFDS exports into JSONL windows for low-level MEM training.

    DROID and Bridge V2 are commonly distributed in RLDS format. This builder expects the
    dataset to be available locally through `tensorflow_datasets.load`.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        split: str,
        *,
        max_episodes: int | None = None,
        frame_horizon: int = 6,
        action_horizon: int = 16,
        action_dim: int = 14,
        frame_key: str | None = None,
        instruction_key: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.split = split
        self.max_episodes = max_episodes
        self.frame_horizon = frame_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.frame_key = frame_key
        self.instruction_key = instruction_key

    def build(self) -> list[LowLevelTrainingRecord]:
        try:
            import tensorflow as tf  # type: ignore
            import tensorflow_datasets as tfds  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "tensorflow and tensorflow_datasets are required to load DROID/Bridge V2 RLDS exports"
            ) from exc
        ds = tfds.load(self.dataset_name, data_dir=self.data_dir, split=self.split)
        records: list[LowLevelTrainingRecord] = []
        for episode_index, episode in enumerate(tfds.as_numpy(ds)):
            if self.max_episodes is not None and episode_index >= self.max_episodes:
                break
            steps = episode["steps"]
            records.extend(self._episode_to_windows(episode_index, steps, tf=tf))
        return records

    def dump_jsonl(self, output_path: str | Path) -> None:
        write_jsonl(output_path, [row.model_dump() for row in self.build()])

    def _episode_to_windows(self, episode_index: int, steps: Any, *, tf: Any) -> list[LowLevelTrainingRecord]:
        observations = steps["observation"]
        actions = steps["action"]
        instructions = self._extract_instruction_stream(steps, observations)
        frame_stream = self._extract_frame_stream(observations)
        proprio_stream = self._extract_proprio_stream(observations)
        action_stream = self._extract_action_stream(actions)
        windows: list[LowLevelTrainingRecord] = []
        max_start = max(len(action_stream) - self.action_horizon + 1, 0)
        for start in range(max_start):
            frame_paths = [str(item) for item in frame_stream[start : start + self.frame_horizon]]
            if not frame_paths:
                continue
            proprio = [list(map(float, item)) for item in proprio_stream[start : start + self.frame_horizon]]
            action_chunk = [list(map(float, item)) for item in action_stream[start : start + self.action_horizon]]
            if len(action_chunk) < self.action_horizon:
                continue
            subtask = instructions[start] if start < len(instructions) else instructions[-1]
            windows.append(
                LowLevelTrainingRecord(
                    episode_id=f"{self.dataset_name}_{episode_index}",
                    goal=instructions[0] if instructions else f"episode_{episode_index}",
                    subtask=subtask,
                    frame_paths=frame_paths,
                    proprio=proprio,
                    action_chunk=action_chunk,
                )
            )
        return windows

    def _extract_instruction_stream(self, steps: Any, observations: Any) -> list[str]:
        key_candidates = [
            self.instruction_key,
            "language_instruction",
            "natural_language_instruction",
            "language_embedding_text",
            "task_instruction",
        ]
        for key in key_candidates:
            if not key:
                continue
            stream = self._maybe_decode_array(steps.get(key)) if hasattr(steps, "get") else None
            if stream is None and hasattr(observations, "get"):
                stream = self._maybe_decode_array(observations.get(key))
            if isinstance(stream, list) and stream:
                return [str(item) for item in stream]
        return ["follow the demonstration"] * len(self._extract_action_stream(steps["action"]))

    def _extract_frame_stream(self, observations: Any) -> list[str]:
        key_candidates = [self.frame_key, "image", "rgb", "wrist_image", "exterior_image_1_left"]
        for key in key_candidates:
            if not key or not hasattr(observations, "get"):
                continue
            value = observations.get(key)
            if value is None:
                continue
            decoded = self._maybe_decode_array(value)
            if decoded:
                return [self._materialize_frame(item, key, index) for index, item in enumerate(decoded)]
        raise ValueError("Could not find image stream in RLDS observation. Set data.frame_key explicitly.")

    def _extract_proprio_stream(self, observations: Any) -> list[list[float]]:
        key_candidates = ["proprio", "state", "robot_state", "joint_position", "cartesian_position"]
        for key in key_candidates:
            if not hasattr(observations, "get"):
                continue
            value = observations.get(key)
            if value is None:
                continue
            decoded = self._maybe_decode_array(value)
            if decoded:
                return [self._pad_feature(list(map(float, item))) for item in decoded]
        return [[0.0] * self.action_dim]

    def _extract_action_stream(self, actions: Any) -> list[list[float]]:
        decoded = self._maybe_decode_array(actions)
        if not decoded:
            return []
        return [self._pad_feature(list(map(float, item))) for item in decoded]

    def _pad_feature(self, values: list[float]) -> list[float]:
        if len(values) >= self.action_dim:
            return values[: self.action_dim]
        return values + [0.0] * (self.action_dim - len(values))

    def _maybe_decode_array(self, value: Any) -> list[Any] | None:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return None

    def _materialize_frame(self, item: Any, key: str, index: int) -> str:
        output_dir = Path(self.data_dir) / "open_pi_mem_cache" / self.dataset_name / key
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"frame_{index:06d}.png"
        if not path.exists():
            from PIL import Image
            import numpy as np

            array = item
            if hasattr(array, "numpy"):
                array = array.numpy()
            array = np.asarray(array)
            Image.fromarray(array.astype("uint8")).save(path)
        return str(path)
