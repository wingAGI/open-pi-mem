from __future__ import annotations

import argparse
from pathlib import Path

from open_pi_mem.data.build_high_level_dataset import build_memory_supervision
from open_pi_mem.data.memory_generation import LLMProviderConfig


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate high-level memory supervision from manual annotation JSON/JSONL files.",
    )
    parser.add_argument(
        "--input",
        default=str(repo_root / "data" / "manual_annotations" / "episodes"),
        help="Input path: a single .json, a .jsonl, or a directory of episode .json files.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "data" / "manual_annotations" / "memory_supervision.manual.jsonl"),
    )
    parser.add_argument(
        "--prompt",
        default=str(repo_root / "prompts" / "memory_summary_prompt.txt"),
    )
    parser.add_argument("--provider", default="openai_compatible")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-mode", default="auto")
    parser.add_argument("--reasoning-split", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=1.5)
    parser.add_argument("--segmentation-mode", default="auto")
    parser.add_argument("--success-mode", default="auto")
    args = parser.parse_args()
    reasoning_split = None
    if args.reasoning_split != "auto":
        reasoning_split = args.reasoning_split == "true"

    build_memory_supervision(
        input_path=args.input,
        output_path=args.output,
        prompt_path=args.prompt,
        llm_config=LLMProviderConfig(
            provider=args.provider,
            model_name=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            api_mode=args.api_mode,
            reasoning_split=reasoning_split,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_backoff_sec=args.retry_backoff_sec,
        ),
        segmentation_mode=args.segmentation_mode,
        success_mode=args.success_mode,
    )
    print(f"Wrote manual high-level supervision to {args.output}")


if __name__ == "__main__":
    main()
