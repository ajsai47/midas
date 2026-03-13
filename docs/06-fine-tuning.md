# Step 5b: Fine-Tune Your Own Model

API-based generation uses a general-purpose model with your scoring formula as
context. Fine-tuning goes further: it bakes your writing style, topics, and
patterns directly into the model weights.

## When to Fine-Tune vs. Just Use the API

```
Use API only if:
  - You have < 100 posts
  - Your writing style is straightforward
  - You are happy with Claude/GPT output after editing
  - You do not want to manage infrastructure

Fine-tune if:
  - You have 200+ posts with engagement data
  - You have a distinctive voice the API does not capture
  - You post frequently (5+ times/week)
  - You want lower latency and no per-token cost
  - You want a model that improves from your edits (DPO)
```

## The SFT -> DPO Pipeline

MIDAS uses a two-stage training pipeline:

**Stage 1: SFT (Supervised Fine-Tuning)**
Train the model to write like you by learning from your highest-performing posts.

**Stage 2: DPO (Direct Preference Optimization)**
Train the model to prefer your edits over its own first drafts, using before/after
pairs from the feedback loop.

```
Your posts (200+)
    |
    v
  [Filter: top 25% by engagement]
    |
    v
  SFT training data (50-100 posts)
    |
    v
  Fine-tuned model (writes like you)
    |
    v
  Generate drafts → you edit → log edits
    |
    v
  DPO training pairs (50+)
    |
    v
  DPO-refined model (writes like you, avoids your corrections)
```

## Data Requirements

| Stage | Minimum | Recommended | Format |
|-------|---------|-------------|--------|
| SFT | 50 posts | 100-200 | JSONL with `prompt` + `completion` |
| DPO | 30 pairs | 50-100 | JSONL with `prompt` + `chosen` + `rejected` |

## Step 1: Prepare SFT Data

Filter your posts to the top performers. The training scripts are standalone files
in the `training/` directory, run directly with Python:

```bash
python training/prepare_sft.py posts.jsonl --output sft_data.jsonl \
  --top-percentile 25 \
  --min-chars 200
```

This generates training examples in chat format:

```json
{
  "prompt": "Write a LinkedIn post about building developer tools",
  "completion": "I spent 6 months building a tool nobody asked for.\n\nHere's why I'd do it again →\n\n..."
}
```

The prompt is auto-generated from the post content (topic extraction). The
completion is your actual post text.

## Step 2: Run SFT Training

Install training dependencies:

```bash
pip install "midas-linkedin[training]"
```

Run training:

```bash
python training/train_sft.py sft_data.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output ./checkpoints/sft-v1 \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --lora-rank 32 \
  --max-length 1024
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `--output` | `./checkpoints/sft` | Output directory |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--learning-rate` | `1e-4` | Learning rate |
| `--lora-rank` | `32` | LoRA rank (higher = more capacity, more VRAM) |
| `--max-length` | `1024` | Max sequence length in tokens |

Training uses LoRA (Low-Rank Adaptation) so you only train a small adapter on
top of the base model. This keeps VRAM requirements manageable.

## Step 3: Prepare DPO Data

After using the fine-tuned model and logging your edits (see
[Step 6](07-feedback-loop.md)), you can export DPO pairs from your feedback log
using the MIDAS CLI:

```bash
midas feedback --export-dpo dpo_data.jsonl --log feedback.jsonl
```

Then prepare the data for training:

```bash
python training/prepare_dpo.py dpo_data.jsonl --output dpo_training_data.jsonl
```

Each feedback entry becomes a DPO pair:

```json
{
  "prompt": "Write a LinkedIn post about...",
  "chosen": "Your edited version (what you actually posted)",
  "rejected": "The model's original draft (before your edits)"
}
```

## Step 4: Run DPO Training

```bash
python training/train_dpo.py dpo_training_data.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --sft-checkpoint ./checkpoints/sft-v1 \
  --output ./checkpoints/dpo-v1 \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --beta 0.1
```

The `--beta` parameter controls how strongly the model moves toward your
preferences. Lower beta = stronger preference learning but risk of overfitting.
Start with 0.1.

## Hardware Requirements

| Model Size | SFT VRAM | DPO VRAM | Estimated Time |
|------------|----------|----------|----------------|
| 3B | 12 GB | 16 GB | 15-30 min |
| 7B | 24 GB | 32 GB | 30-60 min |
| 13B | 40 GB | 48 GB | 1-2 hours |

**Cloud GPU options:**

| Provider | GPU | VRAM | Approximate Cost |
|----------|-----|------|-----------------|
| RunPod | A100 40GB | 40 GB | $1.50/hr |
| Lambda | A100 80GB | 80 GB | $2.00/hr |
| Vast.ai | RTX 4090 | 24 GB | $0.40/hr |
| Modal | A100 40GB | 40 GB | $1.10/hr (per-second) |

For a 7B model, a single RTX 4090 (24 GB) handles SFT. DPO needs an A100 40GB
or gradient checkpointing on a 4090.

Alternatively, use a managed fine-tuning API (OpenAI, Together, Fireworks) and
skip infrastructure management. The training scripts can export data in their
expected formats -- see the `--format` flag on each prepare script.

## Step 5: Use the Fine-Tuned Model

Serve the model locally (vLLM recommended):

```bash
vllm serve ./checkpoints/dpo-v1 --port 8000
```

Then use it with MIDAS:

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
midas draft "topic" --provider local --model dpo-v1 --config my_config.yaml
```

The fine-tuned model plus your scoring formula gives you the best of both worlds:
the model captures your voice and style, the scorer catches structural patterns.

## Expected Improvements

| Approach | Typical Score | Voice Match | Edit Time |
|----------|--------------|-------------|-----------|
| API only (no fine-tune) | 500-700 | Low-medium | 15-20 min |
| SFT fine-tuned | 600-800 | Medium-high | 5-10 min |
| SFT + DPO | 700-900 | High | 2-5 min |

The biggest gain is not the score -- it is edit time. A DPO-trained model produces
drafts that need minimal editing because it has learned your corrections.

## Running the Scripts Programmatically

The training scripts are standalone and not importable as a `midas.training`
package. Run them directly from the command line as shown above, or invoke them
as subprocesses:

```python
import subprocess

# Prepare SFT data
subprocess.run(["python", "training/prepare_sft.py", "posts.jsonl",
                 "--output", "sft_data.jsonl",
                 "--top-percentile", "25",
                 "--min-chars", "200"], check=True)

# Train SFT
subprocess.run(["python", "training/train_sft.py", "sft_data.jsonl",
                 "--base-model", "Qwen/Qwen2.5-7B-Instruct",
                 "--output", "./checkpoints/sft-v1",
                 "--epochs", "3",
                 "--lora-rank", "32"], check=True)

# Export DPO pairs from feedback log (uses the midas CLI)
subprocess.run(["midas", "feedback", "--export-dpo", "dpo_data.jsonl",
                 "--log", "feedback.jsonl"], check=True)

# Prepare DPO training data
subprocess.run(["python", "training/prepare_dpo.py", "dpo_data.jsonl",
                 "--output", "dpo_training_data.jsonl"], check=True)

# Train DPO
subprocess.run(["python", "training/train_dpo.py", "dpo_training_data.jsonl",
                 "--base-model", "Qwen/Qwen2.5-7B-Instruct",
                 "--sft-checkpoint", "./checkpoints/sft-v1",
                 "--output", "./checkpoints/dpo-v1",
                 "--beta", "0.1"], check=True)
```

## Next Step

To start collecting the edit data you need for DPO, continue to
[Step 6: Close the Loop](07-feedback-loop.md).
