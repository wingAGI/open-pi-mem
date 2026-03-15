# open-pi-mem

An open reproduction of **MEM: Multi-Scale Embodied Memory for Vision Language Action Models**.

A practical two-level hierarchical robot policy: plan what to do next (high-level) + execute action sequences (low-level).

## Quick Start

```bash
# Install
pip install torch transformers datasets pyyaml

# Test locally (no data needed)
python scripts/generate_memory_data.py \
  --input examples/episodes.sample.jsonl \
  --output memory_supervision.jsonl \
  --provider rule_based

# Train high-level policy
python scripts/train_high_level.py --config configs/high_level_vlm.yaml

# Test inference
python scripts/run_high_level_inference.py \
  --config configs/high_level_vlm.yaml \
  --image examples/frames/frame_0.png \
  --goal "open the fridge" \
  --prev-memory "" \
  --history-item "reached kitchen | status=success"
```

## Architecture

### High-Level Policy
- **Model:** Qwen/Qwen2.5-VL-3B-Instruct (vision-language model)
- **Input:** current image, goal, past memory, subtask history
- **Output:** structured text `<subtask>...</subtask><memory>...</memory>`
- **Training:** causal language modeling on structured targets

### Low-Level Policy
- **Text:** Gemma-2-2b-it encoder → hidden states
- **Vision:** SigLIP-base-patch16-224 + MEM temporal attention layers
- **Fusion:** text + vision + proprioception → MLP → action predictions
- **Outputs:**
  - Action chunk regression (MSE loss)
  - FAST token head (optional, cross-entropy)
  - Flow matching auxiliary (optional, smooth L1)

## Training

### High-Level

```bash
python scripts/train_high_level.py --config configs/high_level_vlm.yaml
```

**Config:** `configs/high_level_vlm.yaml`
- Model: Qwen/Qwen2.5-VL-3B-Instruct
- Learning rate: 2.0e-5
- Batch size: 1
- Precision: bf16

**Data format (JSONL):**
```json
{
  "goal": "pick up the mug",
  "previous_memory": "found mug in sink",
  "history": [
    {"subtask": "reach sink", "status": "success"},
    {"subtask": "grasp mug", "status": "unknown"}
  ],
  "subtask": "lift mug",
  "memory": "mug lifted from sink"
}
```

### Low-Level

```bash
python scripts/train_low_level.py --config configs/low_level.yaml
```

**Config:** `configs/low_level.yaml`
- Text model: google/gemma-2-2b-it
- Vision model: google/siglip-base-patch16-224
- Action dim: 14 (robot-specific)
- Action horizon: 16 (predict 16 steps ahead)
- Learning rate: 1.0e-4
- Batch size: 1

**Data format (JSONL):**
```json
{
  "instruction": "pick up the mug",
  "frames": ["frame_0.png", "frame_1.png", ...],
  "proprio": [[a1, a2, ...], ...],
  "actions": [[a1, a2, ...], ...]
}
```

## Data Preparation

### High-Level Supervision

**From raw episodes (automatic generation):**

```bash
# Using local rules (no API)
python scripts/generate_memory_data.py \
  --input episodes.jsonl \
  --output memory_supervision.jsonl \
  --provider rule_based

# Using OpenAI-compatible API
export OPENAI_API_KEY=...
python scripts/generate_memory_data.py \
  --input episodes.jsonl \
  --output memory_supervision.jsonl \
  --provider openai_compatible \
  --model gpt-4-mini
```

**Manual annotation (web UI):**

```bash
python scripts/run_annotation_app.py --port 8765
# Open http://127.0.0.1:8765
```

Then export as memory supervision:
```bash
python scripts/generate_manual_high_level_data.py \
  --input data/manual_annotations/episodes \
  --output memory_supervision.manual.jsonl \
  --provider openai_compatible \
  --model gpt-4-mini
```

### Low-Level Data (Robot Trajectories)

**From DROID or Bridge V2 datasets:**

```bash
# Requires local TFDS export (see DROID/Bridge V2 docs)
python scripts/prepare_open_dataset.py \
  --dataset droid \
  --data-dir /path/to/tfds \
  --output low_level_rollouts.jsonl
```

## Inference

### High-Level Planning

```bash
python scripts/run_high_level_inference.py \
  --config configs/high_level_vlm.yaml \
  --image /path/to/frame.jpg \
  --goal "put the milk on the counter" \
  --prev-memory "opened the fridge" \
  --history-item "reach for fridge handle | status=success" \
  --history-item "pull fridge door | status=success"
```

### Low-Level Execution

Currently training-only. To add inference:
- Load checkpoint: `model.load_state_dict(torch.load(ckpt))`
- Forward pass: `outputs = model(input_ids, video, proprio)`
- Extract actions: `outputs["action_chunk"]`

## Configuration

Key settings in YAML configs:

**High-level (`high_level_vlm.yaml`):**
```yaml
model:
  multimodal_backbone_name: Qwen/Qwen2.5-VL-3B-Instruct
  freeze_text_backbone: false
data:
  train_jsonl: memory_supervision.jsonl
  max_total_tokens: 1024
trainer:
  learning_rate: 2.0e-5
  batch_size: 1
```

**Low-level (`low_level.yaml`):**
```yaml
model:
  backbone_name: google/gemma-2-2b-it
  vision_tower_name: google/siglip-base-patch16-224
  action_dim: 14
  action_chunk_horizon: 16
  use_fast_head: true
loss:
  action_mse_weight: 1.0
  fast_token_weight: 1.0
  flow_matching_weight: 1.0
```

## Repository Structure

```
open-pi-mem/
├── configs/                 # YAML experiment configs
├── scripts/                 # Training and data scripts
├── src/open_pi_mem/
│   ├── models/             # Policy architectures
│   ├── training/           # Trainers and loss functions
│   ├── data/               # Dataset loaders and adapters
│   └── utils/              # Config and I/O utilities
├── examples/               # Sample data for testing
├── docs/                   # Design notes
└── README.md
```

## Design Choices

See `docs/design.md` for rationale. Key decisions:

- **Separate backbones:** high and low policies have independent text/vision encoders
- **VLM initialization:** both policies initialized from pretrained VLMs (not random)
- **Structured high-level output:** causal LM on `<subtask>...</subtask><memory>...</memory>` (not classification)
- **Detached action expert:** gradients from action prediction don't flow into vision/text backbone
- **Multi-loss training:** action MSE + optional FAST tokens + optional flow matching

## Testing

**Smoke test with included examples:**

```bash
# ~3 minutes total, validates the full pipeline
python scripts/generate_memory_data.py \
  --input examples/episodes.sample.jsonl \
  --output examples/memory_supervision.sample.jsonl \
  --provider rule_based

python scripts/train_high_level.py --config configs/high_level_vlm.yaml
python scripts/train_low_level.py --config configs/low_level.yaml
```

This verifies:
- Data loading and preprocessing
- Model initialization
- Training loop (forward pass, backward pass, optimization)
- Schema validation

Does not verify:
- Real-world task success (need real robot data)
- Convergence or meaningful learning (10 training steps is too few)

## Not Yet Implemented

- **Low-level inference script** (just training for now)
- **Multi-stage/joint training** (separate training only)
- **Distributed training** (FSDP/DDP)
- **Evaluation metrics** (success rate, trajectory metrics)
- **Real-time robot control** (policy only, no hardware integration)

## Citation

```bibtex
@article{nasiriany2024mem,
  title={MEM: Multi-Scale Embodied Memory for Vision Language Action Models},
  author={Nasiriany, Sameh and others},
  journal={arXiv preprint arXiv:2407.09762},
  year={2024}
}
```

## License

MIT
