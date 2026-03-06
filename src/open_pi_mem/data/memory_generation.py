from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from open_pi_mem.data.schemas import EpisodeRecord, MemoryTrainingRecord, SubtaskEvent


@dataclass
class LLMProviderConfig:
    provider: str = "rule_based"
    model_name: str = "gpt-4.1-mini"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 256
    max_retries: int = 3
    retry_backoff_sec: float = 1.5
    api_mode: str = "auto"
    reasoning_split: bool | None = None


class LLMClient:
    def generate_json(self, prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, config: LLMProviderConfig) -> None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {config.api_key_env} is required for provider={config.provider}")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=config.base_url)
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.max_retries = max(config.max_retries, 1)
        self.retry_backoff_sec = max(config.retry_backoff_sec, 0.0)
        self.api_mode = _resolve_api_mode(config)
        self.reasoning_split = _resolve_reasoning_split(config)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                text = self._generate_text(prompt)
                return _parse_json_payload(text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_sec * attempt)
        assert last_error is not None
        raise RuntimeError(f"LLM generation failed after {self.max_retries} attempts") from last_error

    def _generate_text(self, prompt: str) -> str:
        if self.api_mode == "chat":
            request_kwargs: dict[str, Any] = {}
            if self.reasoning_split is not None:
                request_kwargs["extra_body"] = {"reasoning_split": self.reasoning_split}
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **request_kwargs,
            )
            return response.choices[0].message.content or ""

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        return response.output_text


class RuleBasedFallbackClient(LLMClient):
    def generate_json(self, prompt: str) -> dict[str, Any]:
        lines = [line.strip() for line in prompt.splitlines() if line.strip().startswith("- ")]
        successful = [line[2:].strip() for line in lines if "status=success" in line]
        failures = [line[2:].strip() for line in lines if "status=failure" in line]
        next_subtask = successful[-1] if successful else "continue current task"
        if failures:
            next_subtask = f"recover from failure: {failures[-1].split('|', 1)[0].strip()}"
        memory = [item.split("|", 1)[0].strip() for item in successful[-3:]]
        return {
            "next_subtask": next_subtask,
            "next_memory": "; ".join(memory) if memory else "No stable memory yet.",
        }


class MemorySupervisionBuilder:
    def __init__(self, prompt_template: str, llm_client: LLMClient) -> None:
        self.prompt_template = prompt_template
        self.llm_client = llm_client

    def _build_prompt(self, episode: EpisodeRecord, upto_index: int, prev_memory: str) -> str:
        history_lines = []
        for subtask in episode.subtasks[: upto_index + 1]:
            span = ""
            if subtask.start_index is not None or subtask.end_index is not None:
                span = f" | frames={subtask.start_index}:{subtask.end_index}"
            history_lines.append(f"- {subtask.text} | status={subtask.status}{span}")
        history = "\n".join(history_lines)
        return (
            f"{self.prompt_template}\n\n"
            f"Goal: {episode.goal}\n"
            f"Previous memory: {prev_memory or 'None'}\n"
            f"History:\n{history}\n"
            "Return JSON with keys next_subtask and next_memory."
        )

    def build_records(self, episode: EpisodeRecord) -> list[MemoryTrainingRecord]:
        records: list[MemoryTrainingRecord] = []
        prev_memory = ""
        for idx, _ in enumerate(episode.subtasks):
            prompt = self._build_prompt(episode, idx, prev_memory)
            response = self.llm_client.generate_json(prompt)
            record = MemoryTrainingRecord(
                episode_id=episode.episode_id,
                goal=episode.goal,
                observation_ref=episode.frames[idx] if idx < len(episode.frames) else None,
                prev_memory=prev_memory,
                next_subtask=response["next_subtask"],
                next_memory=response["next_memory"],
                history=[s.model_dump() for s in episode.subtasks[: idx + 1]],
            )
            records.append(record)
            prev_memory = record.next_memory
        return records


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_llm_client(config: LLMProviderConfig) -> LLMClient:
    if config.provider in {"openai", "openai_compatible", "vllm"}:
        return OpenAICompatibleClient(config)
    return RuleBasedFallbackClient()


def _resolve_api_mode(config: LLMProviderConfig) -> str:
    if config.api_mode in {"chat", "responses"}:
        return config.api_mode
    provider = (config.provider or "").lower()
    base_url = (config.base_url or "").lower()
    model_name = (config.model_name or "").lower()
    if "minimax" in provider or "minimax" in base_url or model_name.startswith("minimax"):
        return "chat"
    return "responses"


def _resolve_reasoning_split(config: LLMProviderConfig) -> bool | None:
    if config.reasoning_split is not None:
        return config.reasoning_split
    provider = (config.provider or "").lower()
    base_url = (config.base_url or "").lower()
    model_name = (config.model_name or "").lower()
    if "minimax" in provider or "minimax" in base_url or model_name.startswith("minimax"):
        return True
    return None


def ensure_episode_annotations(
    episode: EpisodeRecord,
    *,
    segmentation_mode: str = "auto",
    success_mode: str = "auto",
) -> EpisodeRecord:
    subtasks = segment_episode(episode, segmentation_mode=segmentation_mode)
    annotated = annotate_subtask_status(episode, subtasks, success_mode=success_mode)
    episode.subtasks = annotated
    return episode


def segment_episode(episode: EpisodeRecord, *, segmentation_mode: str = "auto") -> list[SubtaskEvent]:
    if episode.subtasks:
        return episode.subtasks
    metadata = episode.metadata or {}
    events = metadata.get("subtask_events") or metadata.get("events") or metadata.get("annotations")
    if isinstance(events, list) and events:
        subtasks: list[SubtaskEvent] = []
        for idx, event in enumerate(events):
            text = event.get("text") or event.get("instruction") or event.get("name") or f"subtask_{idx}"
            subtasks.append(
                SubtaskEvent(
                    text=text,
                    status=event.get("status", "unknown"),
                    start_index=event.get("start_index", idx),
                    end_index=event.get("end_index", idx),
                )
            )
        return subtasks
    instructions = _normalize_instruction_stream(metadata)
    if instructions:
        subtasks = []
        start = 0
        prev = instructions[0]
        for idx, current in enumerate(instructions[1:], start=1):
            if current != prev:
                subtasks.append(SubtaskEvent(text=prev, status="unknown", start_index=start, end_index=idx - 1))
                start = idx
                prev = current
        subtasks.append(SubtaskEvent(text=prev, status="unknown", start_index=start, end_index=len(instructions) - 1))
        return subtasks
    return [SubtaskEvent(text=episode.goal, status="unknown", start_index=0, end_index=max(len(episode.frames) - 1, 0))]


def annotate_subtask_status(
    episode: EpisodeRecord,
    subtasks: Iterable[SubtaskEvent],
    *,
    success_mode: str = "auto",
) -> list[SubtaskEvent]:
    metadata = episode.metadata or {}
    success_indices = set(metadata.get("success_subtask_indices", []))
    failure_indices = set(metadata.get("failure_subtask_indices", []))
    terminal_success = metadata.get("terminal_success")
    rewards = metadata.get("rewards") or []
    annotated: list[SubtaskEvent] = []
    subtask_list = list(subtasks)
    for idx, subtask in enumerate(subtask_list):
        if subtask.status != "unknown":
            annotated.append(subtask)
            continue
        status = "unknown"
        if idx in success_indices:
            status = "success"
        elif idx in failure_indices:
            status = "failure"
        elif rewards and subtask.end_index is not None and subtask.end_index < len(rewards):
            reward = rewards[subtask.end_index]
            if reward > 0:
                status = "success"
            elif reward < 0:
                status = "failure"
        elif terminal_success is not None and idx == len(subtask_list) - 1:
            status = "success" if terminal_success else "failure"
        annotated.append(subtask.model_copy(update={"status": status}))
    return annotated


def _normalize_instruction_stream(metadata: dict[str, Any]) -> list[str]:
    candidates = [
        metadata.get("language_instruction_per_step"),
        metadata.get("instruction_per_step"),
        metadata.get("subtask_per_step"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            return [str(item).strip() for item in candidate]
    return []


def _parse_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        think_start = text.find("<think>")
        think_end = text.find("</think>")
        if think_start != -1 and think_end != -1 and think_end > think_start:
            text = text[:think_start] + text[think_end + len("</think>") :]
            text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    payload = json.loads(text)
    if "next_subtask" not in payload or "next_memory" not in payload:
        raise ValueError("LLM response must include next_subtask and next_memory")
    return payload
