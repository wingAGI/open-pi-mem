# open-pi-mem

An open reproduction framework for `MEM: Multi-Scale Embodied Memory for Vision Language Action Models`.

This repo is not a claim of exact PI reproduction. It is a practical engineering baseline for rebuilding the method around open components and explicit assumptions.

## What Changed In This Revision

- real `transformers` backbones replace the previous `Tiny*` placeholders
- Gemma-compatible causal LM loading is wired through `AutoModelForCausalLM`
- SigLIP-compatible vision loading is wired through `AutoModel` + `AutoImageProcessor`
- optional VLM checkpoint import is supported for text tower, vision tower, or a combined checkpoint
- high-level memory data generation can call a real OpenAI-compatible LLM provider
- episode segmentation and success/failure annotation now have explicit preprocessing logic
- low-level training now uses real `Dataset` / `DataLoader` code
- a minimal RLDS-to-JSONL path is added for DROID / Bridge V2 style data

## Scope

The paper leaves several implementation details unspecified: exact parameter sharing, exact loss weighting, full data mixture, and the internal training recipe. This repo makes those choices explicit in config instead of hiding them.

## Architecture

### High-level policy

- backbone: Gemma-style causal LM
- supervision format: causal LM over a structured target
- target format:

```text
<subtask>pick up the mug</subtask>
<memory>mug moved from sink to counter</memory>
```

- training input includes goal, previous memory, and subtask history
- visual context is scaffolded and can be attached later if you want a full image-conditioned planner

### Low-level policy

- text backbone: Gemma-style causal LM hidden states
- vision tower: SigLIP-style encoder
- video memory: MEM-style temporal attention on patch tokens
- proprio projection + text/video fusion
- outputs:
  - action chunk regression head
  - optional FAST token head
  - detached action expert path to match the paper's statement that action-expert gradients do not flow into the VLM backbone

## Weight Initialization And VLM Import

`configs/high_level.yaml` and `configs/low_level.yaml` now support:

- `backbone_name`: HF model name or local path for the text model
- `vision_tower_name`: HF model name or local path for the vision model
- `text_checkpoint`: optional extra checkpoint just for the text tower
- `vision_checkpoint`: optional extra checkpoint just for the vision tower
- `vlm_checkpoint`: optional combined checkpoint with prefix filtering
- `vlm_text_prefixes` / `vlm_vision_prefixes`: how to split a combined checkpoint into the two towers

That is the hook for "start from a VLM, then continue training MEM".

## Episode Annotation App

A lightweight local web app is included under `web/annotator/` for manually labeling episode videos into MEM-style high-level supervision.

What it supports:

- load one or more local video files in the browser
- first place breakpoints only, then auto-build continuous segments
- label each generated segment with `text`, `status`, `confidence`, and `notes`
- save the current episode or all episodes directly into the repository
- import existing JSON / JSONL annotations back into the UI

Start it locally:

```bash
python scripts/run_annotation_app.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

Export format is aligned with the high-level episode schema used by `scripts/generate_memory_data.py`.

Notes:

- workflow is now breakpoint-first: if breakpoints are `[b1, b2, ...]`, the app generates segments `[0,b1] [b1,b2] ... [bn,duration]`
- local video files stay in the browser; the app writes annotations into `data/manual_annotations/` and records `video_path` as a filename or manually edited relative path
- browser localStorage keeps annotation text, but local file handles are not persisted across refreshes, so reload the video if needed
- the app is intended for high-quality human labeling, not collaborative multi-user annotation

## Stage A Local Smoke Test

A minimal local test set is checked into `examples/`:

- `examples/episodes.sample.jsonl`: 4 high-level episodes
- `examples/low_level_rollouts.sample.jsonl`: 16 low-level windows
- `examples/frames/`: 8 local PNG frames

Generate memory supervision locally without calling an external LLM:

```bash
PYTHONPATH=src python scripts/generate_memory_data.py   --input examples/episodes.sample.jsonl   --output examples/memory_supervision.sample.jsonl   --provider rule_based
```

Prepare configs for the sample files by overriding the dataset paths or editing the YAMLs.

These samples are for pipeline validation only:

- schema validation
- segmentation / success-failure annotation
- image loading
- low-level batch collation

They are not intended as meaningful training data.

## High-Level Memory Supervision Pipeline

Open datasets rarely provide PI-style long-horizon language memory labels. The pipeline here is:

1. load raw episodes from JSONL
2. segment the episode into subtasks
3. infer success / failure labels from metadata or reward heuristics
4. build an LLM prompt from `(goal, previous_memory, history)`
5. call a real provider or a rule-based fallback
6. save JSONL supervision records

The raw episode schema can include either:

- explicit `subtasks`
- `metadata.subtask_events`
- per-step instruction streams such as `metadata.language_instruction_per_step`
- success signals such as `metadata.success_subtask_indices`, `metadata.failure_subtask_indices`, `metadata.terminal_success`, or `metadata.rewards`

### Generate memory labels

Rule-based fallback:

```bash
python scripts/generate_memory_data.py \
  --input episodes.jsonl \
  --output memory_supervision.jsonl \
  --provider rule_based
```

OpenAI-compatible provider:

```bash
export OPENAI_API_KEY=...
python scripts/generate_memory_data.py \
  --input episodes.jsonl \
  --output memory_supervision.jsonl \
  --provider openai_compatible \
  --model gpt-4.1-mini \
  --base-url https://api.openai.com/v1
```

You can point `--base-url` at a vLLM or other OpenAI-compatible server.

## Low-Level Data Path For DROID / Bridge V2

The practical minimal route is:

1. obtain a local RLDS / TFDS export for DROID or Bridge V2
2. convert episodes into fixed windows of frames, proprio, and action chunks
3. save them as `low_level_rollouts.jsonl`
4. train with `scripts/train_low_level.py`

### Prepare windows from RLDS

```bash
python scripts/prepare_open_dataset.py \
  --dataset droid \
  --data-dir /path/to/tfds \
  --split train \
  --output low_level_rollouts.jsonl \
  --max-episodes 8
```

For Bridge V2:

```bash
python scripts/prepare_open_dataset.py \
  --dataset bridge_v2 \
  --data-dir /path/to/tfds \
  --split train \
  --output low_level_rollouts.jsonl \
  --max-episodes 8
```

Notes:

- this path expects local `tensorflow` + `tensorflow_datasets`
- image key and instruction key can vary across exports, so `--frame-key` and `--instruction-key` are exposed
- extracted frames are cached under `open_pi_mem_cache/` inside the dataset root

## Training

### High-level

```bash
python scripts/train_high_level.py \
  --config configs/high_level.yaml
```

### Low-level

```bash
python scripts/train_low_level.py \
  --config configs/low_level.yaml
```

## Repository Layout

```text
open-pi-mem/
  configs/                 Default experiment configs
  docs/                    Design notes
  prompts/                 LLM prompts for memory supervision
  scripts/                 CLI entrypoints
  src/open_pi_mem/
    data/                  Dataset schemas, builders, open-data adapters
    models/                HF backbones, video memory, policies, action expert
    training/              Losses and trainers
    utils/                 Config and I/O helpers
```

## Current Boundaries

This repo still leaves several things to the user:

- exact multi-stage or joint training schedule
- exact action tokenization for a faithful FAST reproduction
- true PI-scale data mixture
- distributed training / FSDP / LoRA recipes
- real-time control integration

What is implemented now is the shortest route to a serious open reproduction attempt instead of a paper-only sketch.
