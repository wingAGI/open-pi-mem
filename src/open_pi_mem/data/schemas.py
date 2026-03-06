from __future__ import annotations

from pydantic import BaseModel, Field


class SubtaskEvent(BaseModel):
    text: str
    status: str = Field(description="success|failure|unknown")
    start_index: int | None = None
    end_index: int | None = None


class EpisodeRecord(BaseModel):
    episode_id: str
    goal: str
    frames: list[str] = Field(default_factory=list)
    proprio: list[list[float]] = Field(default_factory=list)
    subtasks: list[SubtaskEvent] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class MemoryTrainingRecord(BaseModel):
    episode_id: str
    goal: str
    observation_ref: str | None = None
    prev_memory: str
    next_subtask: str
    next_memory: str
    history: list[dict] = Field(default_factory=list)


class LowLevelTrainingRecord(BaseModel):
    episode_id: str
    goal: str
    subtask: str
    frame_paths: list[str]
    proprio: list[list[float]]
    action_chunk: list[list[float]]
    fast_tokens: list[int] | None = None
