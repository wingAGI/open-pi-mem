#!/usr/bin/env python3
"""Run open-pi-mem low-level policy on RMBench tasks.

Supports:
  - M(1) tasks: Single-step, no memory required
  - M(n) tasks: Multi-step, requires memory integration with high-level policy

Usage:
  # Evaluate on M(1) tasks only
  python scripts/run_rmbench_eval.py \
    --model checkpoints/low_level.pt \
    --task_config rmbench_tasks.yaml \
    --task_filter M1

  # Evaluate full pipeline (high + low level with memory)
  python scripts/run_rmbench_eval.py \
    --model checkpoints/low_level.pt \
    --high_level_model checkpoints/high_level.pt \
    --task_config rmbench_tasks.yaml \
    --task_filter Mn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

try:
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
except ImportError:
    print("Warning: gymnasium not available. Install via: pip install gymnasium")


def evaluate_m1_task(
    adapter: Any,
    task_name: str,
    env: Any,
    num_episodes: int = 5,
) -> dict[str, float]:
    """Evaluate single-step M(1) task (no memory required).

    Args:
        adapter: RMBenchAdapter instance
        task_name: Name of the M(1) task
        env: Gymnasium environment
        num_episodes: Number of episodes to run

    Returns:
        Results dict with: success_rate, avg_reward, episodes
    """
    results = {"task": task_name, "type": "M(1)", "episodes": []}

    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        adapter.reset()

        # Extract observation components
        frames = info.get("frames", [obs["image"]])  # List of observations
        proprio = obs.get("state", np.zeros(14))
        goal = info.get("task_goal", task_name)

        # Predict actions
        pred = adapter.predict(
            frames=frames,
            proprio_state=proprio.tolist(),
            goal=goal,
        )

        actions = pred["action_chunk"]
        episode_rewards = 0.0
        episode_success = False

        # Execute predicted actions
        for action_step in actions:
            obs, reward, terminated, truncated, info = env.step(action_step)
            episode_rewards += reward
            if info.get("is_success", False):
                episode_success = True
            if terminated or truncated:
                break

        results["episodes"].append(
            {
                "index": ep_idx,
                "reward": float(episode_rewards),
                "success": episode_success,
                "confidence": float(pred["confidence"]),
            }
        )

    # Aggregate
    successes = sum(1 for ep in results["episodes"] if ep["success"])
    results["success_rate"] = successes / num_episodes
    results["avg_reward"] = np.mean([ep["reward"] for ep in results["episodes"]])
    return results


def evaluate_mn_task(
    adapter: Any,
    high_level_adapter: Any,
    task_name: str,
    env: Any,
    num_episodes: int = 5,
) -> dict[str, float]:
    """Evaluate multi-step M(n) task (memory required).

    Args:
        adapter: Low-level RMBenchAdapter instance
        high_level_adapter: High-level policy adapter
        task_name: Name of the M(n) task
        env: Gymnasium environment
        num_episodes: Number of episodes to run

    Returns:
        Results dict with: success_rate, avg_reward, steps, memory_usage
    """
    results = {
        "task": task_name,
        "type": "M(n)",
        "episodes": [],
    }

    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        adapter.reset()

        goal = info.get("task_goal", task_name)
        episode_rewards = 0.0
        total_steps = 0
        current_memory = ""
        episode_success = False

        # Multi-step loop
        max_steps = 100
        while total_steps < max_steps:
            frames = info.get("frames", [obs["image"]])
            proprio = obs.get("state", np.zeros(14))

            # Get high-level planning (if available)
            current_subtask = None
            if high_level_adapter:
                hl_output = high_level_adapter.generate(
                    goal=goal,
                    prev_memory=current_memory,
                    image=frames[-1] if frames else None,
                    history=adapter.history,
                )
                current_subtask = hl_output.get("subtask", None)

            # Low-level execution with memory context
            pred = adapter.predict(
                frames=frames,
                proprio_state=proprio.tolist(),
                goal=goal,
                current_subtask=current_subtask,
                memory_context=current_memory,
            )

            # Execute one action
            action = pred["action_chunk"][0]  # Take first action in chunk
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards += reward
            total_steps += 1

            # Update memory
            if "memory_update" in pred:
                current_memory = pred["memory_update"]
            else:
                current_memory = f"Step {total_steps}: action executed, confidence {pred['confidence']:.2f}"

            adapter.add_history(
                subtask=current_subtask or "unnamed",
                success=info.get("is_success", False),
                memory=current_memory,
            )

            if info.get("is_success", False):
                episode_success = True
                break
            if terminated or truncated:
                break

        results["episodes"].append(
            {
                "index": ep_idx,
                "reward": float(episode_rewards),
                "success": episode_success,
                "steps": total_steps,
                "final_memory": current_memory,
            }
        )

    # Aggregate
    successes = sum(1 for ep in results["episodes"] if ep["success"])
    results["success_rate"] = successes / num_episodes
    results["avg_reward"] = np.mean([ep["reward"] for ep in results["episodes"]])
    results["avg_steps"] = np.mean([ep["steps"] for ep in results["episodes"]])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate open-pi-mem on RMBench tasks"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to low-level policy checkpoint",
    )
    parser.add_argument(
        "--high_level_model",
        type=str,
        default=None,
        help="Path to high-level policy checkpoint (optional)",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="rmbench_tasks.yaml",
        help="Path to RMBench task configuration",
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        choices=["M1", "Mn", "all"],
        default="all",
        help="Filter tasks by type: M(1), M(n), or all",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Episodes per task",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rmbench_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Import adapter (lazy import to avoid circular deps)
    from open_pi_mem.rmbench.adapter import RMBenchAdapter

    print("=" * 60)
    print("RMBench Evaluation: open-pi-mem Low-Level Policy")
    print("=" * 60)

    # Initialize adapter
    print(f"\n[1/3] Loading low-level policy: {args.model}")
    low_level_adapter = RMBenchAdapter(
        model_path=args.model,
        device=args.device,
        memory_enabled=(args.high_level_model is not None),
    )
    print(f"  ✓ Loaded on {args.device}")

    high_level_adapter = None
    if args.high_level_model:
        print(f"\n[2/3] Loading high-level policy: {args.high_level_model}")
        # TODO: Implement HighLevelAdapter
        print("  ⚠ High-level adapter not yet implemented")

    # Load tasks
    print(f"\n[3/3] Loading task config: {args.task_config}")
    if not Path(args.task_config).exists():
        print(f"  ⚠ Config file not found. Using demo tasks.")
        tasks = _get_demo_tasks()
    else:
        import yaml

        with open(args.task_config) as f:
            tasks = yaml.safe_load(f).get("tasks", [])
    print(f"  ✓ Loaded {len(tasks)} tasks")

    # Evaluate tasks
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    all_results = []
    for task in tqdm(tasks, desc="Running tasks"):
        task_name = task.get("name", "unknown")
        task_type = task.get("type", "unknown")

        # Filter
        if args.task_filter == "M1" and task_type != "M(1)":
            continue
        if args.task_filter == "Mn" and task_type != "M(n)":
            continue

        print(f"\n  Task: {task_name} ({task_type})")

        try:
            # Note: RMBench env creation would require actual env_id
            # For now, this is a template
            env = gym.make(task.get("env_id", "PushCube-v1"))

            if task_type == "M(1)":
                result = evaluate_m1_task(
                    low_level_adapter,
                    task_name,
                    env,
                    num_episodes=args.num_episodes,
                )
            else:
                result = evaluate_mn_task(
                    low_level_adapter,
                    high_level_adapter,
                    task_name,
                    env,
                    num_episodes=args.num_episodes,
                )

            all_results.append(result)
            print(
                f"    Success Rate: {result['success_rate']:.1%} | "
                f"Avg Reward: {result['avg_reward']:.2f}"
            )

            env.close()
        except Exception as e:
            print(f"    ✗ Error: {e}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        m1_results = [r for r in all_results if r.get("type") == "M(1)"]
        mn_results = [r for r in all_results if r.get("type") == "M(n)"]
        if m1_results:
            m1_avg = np.mean([r["success_rate"] for r in m1_results])
            print(f"M(1) tasks: {m1_avg:.1%} avg success")
        if mn_results:
            mn_avg = np.mean([r["success_rate"] for r in mn_results])
            print(f"M(n) tasks: {mn_avg:.1%} avg success")


def _get_demo_tasks() -> list[dict[str, str]]:
    """Return demo tasks for testing without RMBench installed."""
    return [
        {"name": "PushCube", "type": "M(1)", "env_id": "PushCube-v1"},
        {"name": "PickPlace", "type": "M(n)", "env_id": "PickPlace-v1"},
    ]


if __name__ == "__main__":
    main()
