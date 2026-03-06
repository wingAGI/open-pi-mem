from __future__ import annotations

import argparse
from pathlib import Path

from open_pi_mem.data.build_high_level_dataset import build_memory_supervision
from open_pi_mem.data.memory_generation import LLMProviderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=str(Path(__file__).resolve().parents[1] / "prompts" / "memory_summary_prompt.txt"))
    parser.add_argument("--provider", default="rule_based")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--segmentation-mode", default="auto")
    parser.add_argument("--success-mode", default="auto")
    args = parser.parse_args()
    build_memory_supervision(
        args.input,
        args.output,
        args.prompt,
        llm_config=LLMProviderConfig(
            provider=args.provider,
            model_name=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        ),
        segmentation_mode=args.segmentation_mode,
        success_mode=args.success_mode,
    )
    print(f"Wrote memory supervision to {args.output}")


if __name__ == "__main__":
    main()
