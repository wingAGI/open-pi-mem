#!/usr/bin/env python3
"""Example: Running open-pi-mem on RMBench tasks.

This example demonstrates:
1. Loading a trained low-level policy
2. Running inference on RMBench observations
3. Handling both M(1) and M(n) tasks with memory
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Import RMBench adapter
from open_pi_mem.rmbench.adapter import RMBenchAdapter


def example_m1_single_inference() -> None:
    """Example 1: Single inference on M(1) task (no memory)."""
    print("\n" + "=" * 60)
    print("Example 1: M(1) Task - Single Inference")
    print("=" * 60)

    # Simulate RMBench observation
    # In real scenario: obs, info = env.reset()
    frames = [Image.new("RGB", (224, 224), color="blue") for _ in range(5)]
    proprio_state = [0.0] * 14  # 14-DOF robot (identity config)
    goal = "push cube to target"

    # Load model
    adapter = RMBenchAdapter(
        model_path="checkpoints/low_level_best.pt",
        device="cuda",
    )

    # Predict actions
    print(f"\nPredicting actions for goal: '{goal}'")
    result = adapter.predict(
        frames=frames,
        proprio_state=proprio_state,
        goal=goal,
        current_subtask=None,  # No high-level planning
        memory_context=None,  # No memory
    )

    print(f"  Action shape: {result['action_chunk'].shape}")  # (16, 14)
    print(f"  Prediction confidence: {result['confidence']:.2%}")
    print(f"  First 3 actions:")
    for i, action in enumerate(result["action_chunk"][:3]):
        print(f"    Step {i+1}: {action}")


def example_m1_episode() -> None:
    """Example 2: Full episode rollout on M(1) task."""
    print("\n" + "=" * 60)
    print("Example 2: M(1) Task - Full Episode")
    print("=" * 60)

    adapter = RMBenchAdapter(
        model_path="checkpoints/low_level_best.pt",
        device="cuda",
    )

    # Simulate RMBench environment
    # In real scenario: env = gym.make("PushCube-v1")
    observations = []
    rewards = []
    success = False

    # Mock: Create synthetic observations (in reality, from env.step())
    goal = "push cube to target"
    num_steps = 20
    current_proprio = [0.0] * 14

    for step in range(num_steps):
        # Get frames (in reality: from env observation)
        frames = [
            Image.new("RGB", (224, 224), color=f"#{step*10:02x}{step*10:02x}{step*10:02x}")
            for _ in range(5)
        ]

        # Predict next actions
        result = adapter.predict(
            frames=frames,
            proprio_state=current_proprio,
            goal=goal,
            current_subtask=None,
            memory_context=None,
        )

        # Execute first action in chunk
        action = result["action_chunk"][0]
        observations.append(frames[-1])
        rewards.append(np.random.random())  # Mock reward

        # Update proprioception (in reality: from env)
        current_proprio = (np.array(current_proprio) + action).tolist()

        print(
            f"Step {step+1:2d}: action mean={action.mean():.3f}, "
            f"reward={rewards[-1]:.3f}, confidence={result['confidence']:.2%}"
        )

        # Simulate task completion
        if np.random.random() < 0.05:  # 5% chance per step
            success = True
            break

    print(f"\nEpisode Summary:")
    print(f"  Steps: {len(rewards)}")
    print(f"  Total reward: {sum(rewards):.2f}")
    print(f"  Success: {success}")


def example_mn_with_memory() -> None:
    """Example 3: M(n) task with memory and history."""
    print("\n" + "=" * 60)
    print("Example 3: M(n) Task - Multi-Step with Memory")
    print("=" * 60)

    adapter = RMBenchAdapter(
        model_path="checkpoints/low_level_best.pt",
        device="cuda",
        memory_enabled=True,
    )
    adapter.reset()

    goal = "pick up cube and place it on the table"
    steps_data = [
        {
            "substep": "Reach for cube",
            "image_color": "ff0000",
            "proprio": [0.5, 0.5, 0.5] + [0.0] * 11,
            "success": True,
        },
        {
            "substep": "Grasp cube",
            "image_color": "00ff00",
            "proprio": [0.5, 0.5, 0.2] + [0.0] * 11,
            "success": True,
        },
        {
            "substep": "Move to table",
            "image_color": "0000ff",
            "proprio": [0.3, 0.8, 0.2] + [0.0] * 11,
            "success": True,
        },
        {
            "substep": "Place on table",
            "image_color": "ffff00",
            "proprio": [0.3, 0.8, 0.7] + [0.0] * 11,
            "success": True,
        },
    ]

    current_memory = ""

    for step_idx, step_data in enumerate(steps_data, 1):
        print(f"\nStep {step_idx}: {step_data['substep']}")

        # Create frames
        frames = [
            Image.new("RGB", (224, 224), color=f"#{step_data['image_color']}")
            for _ in range(5)
        ]

        # Get history context
        history_context = adapter.get_history_context()

        # Predict actions with memory
        result = adapter.predict(
            frames=frames,
            proprio_state=step_data["proprio"],
            goal=goal,
            current_subtask=step_data["substep"],
            memory_context=current_memory,
        )

        print(f"  Memory input: '{current_memory if current_memory else '(start)'}'")
        print(f"  Confidence: {result['confidence']:.2%}")

        # Execute actions
        success = step_data["success"]
        memory_update = result.get("memory_update", f"Executed: {step_data['substep']}")
        current_memory = memory_update

        # Track in history
        adapter.add_history(
            subtask=step_data["substep"],
            success=success,
            memory=memory_update,
        )

        print(f"  Memory output: '{memory_update}'")
        print(f"  Status: {'✓' if success else '✗'}")

    print(f"\nFinal History:")
    for i, entry in enumerate(adapter.history, 1):
        print(
            f"  {i}. {entry['subtask']} "
            f"[{'success' if entry['success'] else 'failed'}] - "
            f"{entry['memory']}"
        )


def example_batch_inference() -> None:
    """Example 4: Batch inference for multiple tasks."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Inference - Multiple Tasks")
    print("=" * 60)

    adapter = RMBenchAdapter(
        model_path="checkpoints/low_level_best.pt",
        device="cuda",
    )

    tasks = [
        {"name": "PushCube", "goal": "push cube to target location"},
        {"name": "PickCube", "goal": "pick up cube and hold it"},
        {"name": "ReachTarget", "goal": "move end-effector to target"},
    ]

    results = []

    for task in tasks:
        frames = [Image.new("RGB", (224, 224), color="gray") for _ in range(5)]
        proprio = [0.0] * 14

        result = adapter.predict(
            frames=frames,
            proprio_state=proprio,
            goal=task["goal"],
        )

        results.append(
            {
                "task": task["name"],
                "goal": task["goal"],
                "action_count": len(result["action_chunk"]),
                "confidence": float(result["confidence"]),
            }
        )

    print("\nResults Summary:")
    print(f"{'Task':<15} {'Goal':<30} {'Actions':<8} {'Confidence':<12}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['task']:<15} {r['goal']:<30} "
            f"{r['action_count']:<8} {r['confidence']:<12.2%}"
        )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("open-pi-mem RMBench Integration Examples")
    print("=" * 60)

    try:
        # Run examples
        example_m1_single_inference()
        example_m1_episode()
        example_mn_with_memory()
        example_batch_inference()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n⚠ Warning: {e}")
        print("Make sure the checkpoint file exists at the specified path.")
        print("Download from: [checkpoint URL]")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
