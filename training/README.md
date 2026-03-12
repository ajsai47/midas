# MIDAS Fine-Tuning Guide

Train a language model to write posts in your personal style using your own performance data.

The pipeline has two stages: **SFT** (Supervised Fine-Tuning) teaches the model your writing style, then **DPO** (Direct Preference Optimization) teaches it to prefer the style that drives higher engagement.

## Prerequisites

**Hardware:**
- GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A100, H100)
- With `--load-in-4bit` (QLoRA): 12GB+ VRAM works (RTX 3060 Ti, T4)
- CPU-only: not recommended (days instead of minutes)

**Cloud options if you don't have a local GPU:**
- [Lambda Cloud](https://lambdalabs.com/) — A100 at ~$1.10/hr
- [RunPod](https://www.runpod.io/) — A100 at ~$0.79/hr
- [Google Colab Pro](https://colab.google/) — A100 at ~$10/month

**Software:**
```bash
pip install midas-linkedin[training]
```

This installs PyTorch, Transformers, TRL, PEFT, Datasets, and Accelerate.

For 4-bit quantization (QLoRA), also install:
```bash
pip install bitsandbytes
```

## Input Data Format

Your post data should be a JSONL file with one post per line:

```json
{"text": "Your post content here...", "reactions": 142, "comments": 23, "reposts": 8, "date": "2025-11-15", "has_image": false}
{"text": "Another post...", "reactions": 87, "comments": 12, "reposts": 3, "date": "2025-11-10", "has_image": true}
```

Required: `text`. Optional: `reactions`, `comments`, `reposts`, `date`, `has_image`.

Export this from your LinkedIn data (use `midas export` or your own scraper).

## Step 1: Prepare SFT Data

Filter to your best posts and convert them into training examples:

```bash
python training/prepare_sft.py \
    --input data/posts.jsonl \
    --output-dir ./data/ \
    --system-prompt "You are a LinkedIn thought leader who writes about AI and community building. Your tone is authentic, direct, and conversational." \
    --top-percentile 0.30 \
    --min-reactions 10 \
    --min-text-length 200 \
    --eval-size 30
```

This produces:
- `data/sft_train.jsonl` — training examples
- `data/sft_eval.jsonl` — held-out evaluation examples

**Tuning tips:**
- ~200-500 training examples is the sweet spot for SFT
- Raise `--min-reactions` if you have lots of data, lower it if you have fewer posts
- The `--system-prompt` defines the model's persona — make it specific to you

## Step 2: Train SFT

```bash
python training/train_sft.py \
    --model Qwen/Qwen3-8B \
    --data ./data/sft_train.jsonl \
    --eval-data ./data/sft_eval.jsonl \
    --output-dir ./checkpoints/sft/ \
    --lora-rank 32 \
    --lr 1e-4 \
    --epochs 3 \
    --batch-size 4 \
    --max-length 2048 \
    --save-steps 50
```

**For limited VRAM (12-16GB):**
```bash
python training/train_sft.py \
    --model Qwen/Qwen3-8B \
    --data ./data/sft_train.jsonl \
    --output-dir ./checkpoints/sft/ \
    --load-in-4bit \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --max-length 1024
```

**Expected training time:**
| Examples | GPU | Time |
|----------|-----|------|
| 200 | A100 (80GB) | ~10 min |
| 200 | RTX 4090 (24GB) | ~20 min |
| 200 | T4 (16GB, 4-bit) | ~45 min |
| 1000 | A100 (80GB) | ~40 min |
| 1000 | RTX 4090 (24GB) | ~90 min |

**What to watch:**
- Loss should drop from ~3.5 to ~1.5-2.5 over training
- If loss plateaus early, try a higher learning rate
- If loss spikes, reduce learning rate or increase warmup

## Step 3: Prepare DPO Data

Create preference pairs by comparing high vs low engagement posts on similar topics:

```bash
python training/prepare_dpo.py \
    --input data/posts.jsonl \
    --output-dir ./data/ \
    --max-pairs 500 \
    --min-engagement-ratio 2.0 \
    --min-chosen-reactions 10 \
    --max-appearances 5 \
    --eval-size 20
```

This produces:
- `data/dpo_train.jsonl` — preference pairs for training
- `data/dpo_eval.jsonl` — held-out pairs for evaluation

**Tuning tips:**
- You need at least ~50 quality pairs for DPO to work
- Lower `--min-engagement-ratio` to 1.5 if you're not getting enough pairs
- `--max-appearances 5` prevents any single post from dominating the training

## Step 4: Train DPO

Start from your SFT checkpoint:

```bash
python training/train_dpo.py \
    --model ./checkpoints/sft/final/ \
    --train-data ./data/dpo_train.jsonl \
    --eval-data ./data/dpo_eval.jsonl \
    --output-dir ./checkpoints/dpo/ \
    --lora-rank 32 \
    --lr 1e-5 \
    --epochs 1 \
    --batch-size 4 \
    --beta 0.1 \
    --save-steps 15
```

**Expected training time:**
| Pairs | GPU | Time |
|-------|-----|------|
| 100 | A100 (80GB) | ~5 min |
| 100 | RTX 4090 (24GB) | ~15 min |
| 500 | A100 (80GB) | ~20 min |
| 500 | RTX 4090 (24GB) | ~60 min |

**What to watch:**
- `rewards/margins` should be positive and increasing (model prefers chosen)
- `rewards/accuracies` should trend toward 0.7+ (above random chance of 0.5)
- If margins go negative, reduce beta or learning rate

## Using Your Fine-Tuned Model

After training, use the DPO checkpoint (or SFT-only if you skipped DPO):

```bash
midas draft --provider local --model ./checkpoints/dpo/final/
```

## Recommended Models

| Model | Size | VRAM (LoRA) | VRAM (QLoRA) | Notes |
|-------|------|-------------|--------------|-------|
| Qwen/Qwen3-8B | 8B | ~20GB | ~8GB | Best all-around, strong instruction following |
| meta-llama/Llama-3.1-8B-Instruct | 8B | ~20GB | ~8GB | Great for English content |
| mistralai/Mistral-7B-Instruct-v0.3 | 7B | ~18GB | ~7GB | Fast, efficient |
| Qwen/Qwen3-4B | 4B | ~12GB | ~5GB | Budget option, still decent quality |
| google/gemma-2-9b-it | 9B | ~22GB | ~9GB | Strong reasoning |

## Troubleshooting

**Out of memory (OOM):**
1. Add `--load-in-4bit` for QLoRA
2. Reduce `--batch-size` to 1
3. Increase `--gradient-accumulation-steps` to compensate
4. Reduce `--max-length` to 1024

**Loss not decreasing:**
- Check your data quality — are the "high-performing" posts actually good?
- Try a higher learning rate (2e-4 for SFT, 5e-5 for DPO)
- Ensure you have enough data (50+ examples for SFT, 50+ pairs for DPO)

**Model outputs are generic / don't match your style:**
- Use a more specific `--system-prompt` in data preparation
- Increase `--top-percentile` to be more selective (e.g., 0.15 for top 15%)
- Train for more epochs (but watch for overfitting)

**DPO makes output worse:**
- Lower beta (try 0.05)
- Ensure your preference pairs have clear quality gaps (`--min-engagement-ratio 3.0`)
- Use the SFT checkpoint directly — DPO is optional
