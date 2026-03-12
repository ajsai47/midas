#!/usr/bin/env python3
"""MIDAS SFT Training — Fine-tune a language model on high-performing posts.

Uses HuggingFace Transformers + TRL + PEFT (LoRA) for parameter-efficient
supervised fine-tuning. Designed for a single GPU with 24GB+ VRAM.

Usage:
    python training/train_sft.py \
        --model Qwen/Qwen3-8B \
        --data ./data/sft_train.jsonl \
        --eval-data ./data/sft_eval.jsonl \
        --output-dir ./checkpoints/sft/ \
        --epochs 3 \
        --batch-size 4 \
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Guard imports so the script gives a clear error if deps are missing
try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer, SFTConfig
except ImportError as exc:
    print(
        f"Missing dependency: {exc}\n"
        "Install training dependencies with: pip install midas-linkedin[training]",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with SFT on high-performing posts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help=(
            "HuggingFace model name or local path. Recommended models:\n"
            "  Qwen/Qwen3-8B       — strong all-around, good instruction following\n"
            "  meta-llama/Llama-3.1-8B-Instruct — good for English content\n"
            "  mistralai/Mistral-7B-Instruct-v0.3 — fast, efficient\n"
            "Default: Qwen/Qwen3-8B"
        ),
    )

    # Data
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to SFT training JSONL (output of prepare_sft.py).",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Path to SFT evaluation JSONL (optional).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints/sft"),
        help="Directory for checkpoints and final model (default: ./checkpoints/sft/).",
    )

    # LoRA hyperparameters
    parser.add_argument(
        "--lora-rank", "-r",
        type=int,
        default=32,
        help=(
            "LoRA rank (default: 32). Controls adapter capacity.\n"
            "  16 = lighter, faster, less expressive\n"
            "  32 = good balance for most fine-tunes\n"
            "  64 = more expressive, needs more data to avoid overfitting"
        ),
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help=(
            "LoRA alpha (default: 64). Scaling factor, typically 2x rank.\n"
            "Higher alpha = stronger adapter influence per step."
        ),
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05). Light regularization to prevent overfitting.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help=(
            "Learning rate (default: 1e-4). Standard for LoRA SFT.\n"
            "  1e-4 = good for most SFT tasks\n"
            "  5e-5 = more conservative, use with larger datasets\n"
            "  2e-4 = aggressive, use with small datasets (<100 examples)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help=(
            "Number of training epochs (default: 3).\n"
            "  2-3 = typical for SFT with 200-2000 examples\n"
            "  1   = use for very large datasets (5000+)\n"
            "  5+  = only for very small datasets (<50), watch for overfitting"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Per-device batch size (default: 4).\n"
            "Effective batch = batch-size * gradient-accumulation-steps.\n"
            "Reduce to 1-2 if running out of VRAM."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help=(
            "Gradient accumulation steps (default: 4).\n"
            "Effective batch = batch-size * this value = 4 * 4 = 16 by default."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help=(
            "Maximum sequence length in tokens (default: 2048).\n"
            "LinkedIn posts rarely exceed 1500 tokens, so 2048 gives headroom.\n"
            "Reduce to 1024 to save VRAM on smaller GPUs."
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save a checkpoint every N steps (default: 50).",
    )

    # Quantization
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help=(
            "Load model in 4-bit quantization (QLoRA). Reduces VRAM from ~18GB to ~8GB\n"
            "for 8B models. Requires bitsandbytes: pip install bitsandbytes"
        ),
    )

    # Misc
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio (default: 0.05). Fraction of total steps for LR warmup.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01). Light L2 regularization.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log metrics every N steps (default: 10).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 mixed precision (default: True). Requires Ampere+ GPU.",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        default=False,
        help="Disable bfloat16 (use fp16 or fp32 instead).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# LoRA target modules per model family
# ---------------------------------------------------------------------------

# Different model architectures name their attention layers differently.
# We target all attention projection matrices + the MLP gate for maximum
# adapter expressiveness without touching embeddings or layer norms.

LORA_TARGET_MODULES: dict[str, list[str]] = {
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
}


def get_target_modules(model_name: str) -> list[str]:
    """Infer LoRA target modules from the model name."""
    name_lower = model_name.lower()
    for family, modules in LORA_TARGET_MODULES.items():
        if family in name_lower:
            return modules
    # Fallback: target attention projections (works for most transformer models)
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    use_bf16 = args.bf16 and not args.no_bf16
    print(f"Configuration:")
    print(f"  Model:          {args.model}")
    print(f"  Data:           {args.data}")
    print(f"  Eval data:      {args.eval_data}")
    print(f"  Output:         {args.output_dir}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size} x {args.gradient_accumulation_steps} (effective: {args.batch_size * args.gradient_accumulation_steps})")
    print(f"  Max length:     {args.max_length}")
    print(f"  4-bit quant:    {args.load_in_4bit}")
    print(f"  BF16:           {use_bf16}")
    print()

    # -----------------------------------------------------------------------
    # 1. Load tokenizer
    # -----------------------------------------------------------------------
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    # Ensure pad token exists (some models like LLaMA don't set one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -----------------------------------------------------------------------
    # 2. Load model (optionally quantized)
    # -----------------------------------------------------------------------
    print("Loading model ...")
    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
        "device_map": "auto",  # Automatically distribute across available GPUs
    }

    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",             # NormalFloat4 — best for LoRA
            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16 for speed
            bnb_4bit_use_double_quant=True,         # Nested quantization saves ~0.4GB
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Disable cache during training (incompatible with gradient checkpointing)
    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # 3. Apply LoRA adapter
    # -----------------------------------------------------------------------
    target_modules = get_target_modules(args.model)
    print(f"Applying LoRA (rank={args.lora_rank}) to: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",  # Don't train biases — keeps adapter small
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # 4. Load dataset
    # -----------------------------------------------------------------------
    print("Loading dataset ...")
    data_files: dict[str, str] = {"train": str(args.data)}
    if args.eval_data and args.eval_data.exists():
        data_files["eval"] = str(args.eval_data)

    dataset = load_dataset("json", data_files=data_files)
    print(f"  Train: {len(dataset['train'])} examples")
    if "eval" in dataset:
        print(f"  Eval:  {len(dataset['eval'])} examples")

    # -----------------------------------------------------------------------
    # 5. Configure training
    # -----------------------------------------------------------------------

    # Determine evaluation strategy
    eval_strategy = "steps" if "eval" in dataset else "no"
    eval_steps = args.save_steps if eval_strategy == "steps" else None

    training_args = SFTConfig(
        output_dir=str(args.output_dir),

        # Core training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_length,

        # Optimizer
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",          # Standard AdamW, good default
        lr_scheduler_type="cosine",   # Cosine decay — smooth LR reduction

        # Precision
        bf16=use_bf16,
        fp16=not use_bf16,

        # Checkpointing & logging
        save_steps=args.save_steps,
        save_total_limit=5,           # Keep only the 5 most recent checkpoints
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,

        # Memory optimization
        gradient_checkpointing=True,  # Trades compute for VRAM (~30% savings)
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Reproducibility
        seed=42,
        data_seed=42,

        # Output
        report_to="none",            # Set to "wandb" or "tensorboard" if desired
        push_to_hub=False,
    )

    # -----------------------------------------------------------------------
    # 6. Initialize trainer
    # -----------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        processing_class=tokenizer,
    )

    # -----------------------------------------------------------------------
    # 7. Train
    # -----------------------------------------------------------------------
    print("\nStarting SFT training ...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)

    # -----------------------------------------------------------------------
    # 8. Save final model
    # -----------------------------------------------------------------------
    final_dir = args.output_dir / "final"
    print(f"\nSaving final model to {final_dir} ...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\nTraining complete!")
    print(f"  Final checkpoint: {final_dir}")
    print(f"  Total steps: {metrics.get('train_steps', 'N/A')}")
    print(f"  Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"\nTo use this model:")
    print(f"  midas draft --provider local --model {final_dir}")


if __name__ == "__main__":
    main()
