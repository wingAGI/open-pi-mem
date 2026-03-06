from __future__ import annotations

import argparse

from open_pi_mem.data.open_datasets import RLDSWindowBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["droid", "bridge_v2"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--frame-horizon", type=int, default=6)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--frame-key", default=None)
    parser.add_argument("--instruction-key", default=None)
    args = parser.parse_args()

    dataset_name = {
        "droid": "droid",
        "bridge_v2": "bridge/1.0.0",
    }[args.dataset]
    builder = RLDSWindowBuilder(
        dataset_name=dataset_name,
        data_dir=args.data_dir,
        split=args.split,
        max_episodes=args.max_episodes,
        frame_horizon=args.frame_horizon,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        frame_key=args.frame_key,
        instruction_key=args.instruction_key,
    )
    builder.dump_jsonl(args.output)
    print(f"Wrote low-level windows to {args.output}")


if __name__ == "__main__":
    main()
