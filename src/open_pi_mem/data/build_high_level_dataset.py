from __future__ import annotations

from pathlib import Path

from open_pi_mem.data.memory_generation import (
    LLMProviderConfig,
    MemorySupervisionBuilder,
    build_llm_client,
    ensure_episode_annotations,
    load_prompt,
)
from open_pi_mem.data.schemas import EpisodeRecord
from open_pi_mem.utils.io import read_jsonl, write_jsonl


def build_memory_supervision(
    input_path: str,
    output_path: str,
    prompt_path: str,
    *,
    llm_config: LLMProviderConfig | None = None,
    segmentation_mode: str = "auto",
    success_mode: str = "auto",
) -> None:
    rows = read_jsonl(input_path)
    prompt = load_prompt(prompt_path)
    llm_client = build_llm_client(llm_config or LLMProviderConfig())
    builder = MemorySupervisionBuilder(prompt_template=prompt, llm_client=llm_client)
    output_rows = []
    for row in rows:
        episode = EpisodeRecord.model_validate(row)
        episode = ensure_episode_annotations(
            episode,
            segmentation_mode=segmentation_mode,
            success_mode=success_mode,
        )
        output_rows.extend(record.model_dump() for record in builder.build_records(episode))
    write_jsonl(output_path, output_rows)


__all__ = ["build_memory_supervision"]
