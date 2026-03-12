#!/usr/bin/env python3
"""MIDAS DPO Training — Align a fine-tuned model to prefer high-engagement content.

Uses HuggingFace TRL's DPOTrainer to train on preference pairs where the
"chosen" response is a high-performing post and the "rejected" response is
a lower-performing post on the same topic.

DPO runs AFTER SFT. It refines the model's style preferences without
catastrophically forgetting the SFT knowledge — which is why we use a lower
learning rate (10x lower than SFT) and fewer epochs (typically 1).

Usage:
    python training/train_dpo.py \
        --model ./checkpoints/sft/final/ \
        --train-data ./data/dpo_train.jsonl \
        --eval-data ./data/dpo_eval.jsonl \
        --output-dir ./checkpoints/dpo/ \
        --epochs 1 \
        --lr 1e-5 \
        --beta 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer
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

# LoRA targets — same as SFT for consistency
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
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO alignment training on preference pairs from posts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "HuggingFace model name or path to SFT checkpoint.\n"
            "Typically: ./checkpoints/sft/final/"
        ),
    )

    # Data
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to DPO training JSONL (output of prepare_dpo.py).",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Path to DPO evaluation JSONL (optional).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints/dpo"),
        help="Directory for checkpoints and final model (default: ./checkpoints/dpo/).",
    )

    # LoRA hyperparameters
    parser.add_argument(
        "--lora-rank", "-r",
        type=int,
        default=32,
        help=(
            "LoRA rank (default: 32). Should match or be <= SFT rank.\n"
            "For DPO, the adapter learns subtle style preferences,\n"
            "so the same rank as SFT works well."
        ),
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha (default: 64). Scaling factor, typically 2x rank.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05).",
    )

    # DPO hyperparameters
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help=(
            "DPO beta parameter (default: 0.1). Controls how strongly the model\n"
            "is pushed toward the chosen response vs the rejected one.\n"
            "  0.1 = standard, balanced preference learning\n"
            "  0.05 = weaker signal, more conservative alignment\n"
            "  0.2 = stronger push, risk of over-correction\n"
            "  0.5 = very aggressive, only use with high-quality pairs"
        ),
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo"],
        help=(
            "DPO loss variant (default: sigmoid).\n"
            "  sigmoid = original DPO loss, most stable\n"
            "  hinge   = margin-based, sometimes better generalization\n"
            "  ipo     = Identity Preference Optimization, regularized variant"
        ),
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help=(
            "Learning rate (default: 1e-5). 10x lower than SFT because DPO\n"
            "is a refinement step — we want to preserve SFT knowledge.\n"
            "  1e-5 = standard for DPO after SFT\n"
            "  5e-6 = more conservative\n"
            "  5e-5 = aggressive, only with many high-quality pairs"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help=(
            "Number of training epochs (default: 1).\n"
            "DPO typically needs only 1 epoch — multiple epochs risk\n"
            "overfitting to the preference signal and forgetting SFT."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Per-device batch size (default: 4).\n"
            "DPO processes pairs (chosen + rejected), so actual VRAM is ~2x SFT.\n"
            "Reduce to 1-2 if running out of memory."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048).",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help=(
            "Maximum prompt length in tokens (default: 512).\n"
            "The prompt is system + user instruction — typically short.\n"
            "Remaining budget goes to chosen/rejected responses."
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=15,
        help=(
            "Save a checkpoint every N steps (default: 15).\n"
            "DPO runs are short, so save frequently."
        ),
    )

    # Quantization
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit quantization (QLoRA). Halves VRAM usage.",
    )

    # Misc
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help=(
            "Warmup ratio (default: 0.1). Slightly higher than SFT because\n"
            "DPO is sensitive to early gradient noise."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Log metrics every N steps (default: 5). More frequent than SFT since DPO runs are shorter.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 mixed precision (default: True).",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        default=False,
        help="Disable bfloat16.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    use_bf16 = args.bf16 and not args.no_bf16
    print(f"Configuration:")
    print(f"  Model:          {args.model}")
    print(f"  Train data:     {args.train_data}")
    print(f"  Eval data:      {args.eval_data}")
    print(f"  Output:         {args.output_dir}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  DPO beta:       {args.beta}")
    print(f"  Loss type:      {args.loss_type}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size} x {args.gradient_accumulation_steps}")
    print(f"  Max length:     {args.max_length} (prompt: {args.max_prompt_length})")
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -----------------------------------------------------------------------
    # 2. Load model
    # -----------------------------------------------------------------------
    print("Loading model ...")
    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
        "device_map": "auto",
    }

    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Check if the model path contains a PEFT adapter (from SFT)
    model_path = Path(args.model)
    is_peft_checkpoint = (model_path / "adapter_config.json").exists()

    if is_peft_checkpoint:
        # Load base model from the adapter config, then merge the SFT adapter
        print(f"  Detected PEFT checkpoint at {args.model}")
        print("  Loading base model and merging SFT adapter ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, **model_kwargs
        )
        # If it's a PEFT model saved via save_pretrained, AutoModelForCausalLM
        # may not auto-load the adapter. Try loading as PeftModel.
        try:
            model = PeftModel.from_pretrained(model, args.model)
            model = model.merge_and_unload()
            print("  SFT adapter merged successfully.")
        except Exception:
            # Model might already be merged or not a PEFT model
            print("  Model loaded directly (no separate adapter to merge).")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # 3. Apply fresh LoRA adapter for DPO
    # -----------------------------------------------------------------------
    target_modules = get_target_modules(args.model)
    print(f"Applying LoRA (rank={args.lora_rank}) to: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    # DPOTrainer applies LoRA itself when given a peft_config, but we need
    # the ref_model to be the un-adapted version. TRL handles this internally
    # when peft_config is provided.

    # -----------------------------------------------------------------------
    # 4. Load dataset
    # -----------------------------------------------------------------------
    print("Loading dataset ...")
    data_files: dict[str, str] = {"train": str(args.train_data)}
    if args.eval_data and args.eval_data.exists():
        data_files["eval"] = str(args.eval_data)

    dataset = load_dataset("json", data_files=data_files)
    print(f"  Train: {len(dataset['train'])} pairs")
    if "eval" in dataset:
        print(f"  Eval:  {len(dataset['eval'])} pairs")

    # -----------------------------------------------------------------------
    # 5. Configure DPO training
    # -----------------------------------------------------------------------

    eval_strategy = "steps" if "eval" in dataset else "no"
    eval_steps = args.save_steps if eval_strategy == "steps" else None

    training_args = DPOConfig(
        output_dir=str(args.output_dir),

        # DPO-specific
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

        # Core training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimizer
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        lr_scheduler_type="cosine",

        # Precision
        bf16=use_bf16,
        fp16=not use_bf16,

        # Checkpointing & logging
        save_steps=args.save_steps,
        save_total_limit=5,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Reproducibility
        seed=42,
        data_seed=42,

        # Output
        report_to="none",
        push_to_hub=False,
    )

    # -----------------------------------------------------------------------
    # 6. Initialize DPO trainer
    # -----------------------------------------------------------------------
    # DPOTrainer internally creates a reference model (frozen copy) when
    # peft_config is provided — no need to create ref_model manually.
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # -----------------------------------------------------------------------
    # 7. Train
    # -----------------------------------------------------------------------
    print("\nStarting DPO training ...")
    print("  (DPO processes chosen+rejected pairs — expect ~2x VRAM vs SFT)")
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

    print("\nDPO training complete!")
    print(f"  Final checkpoint: {final_dir}")
    print(f"  Total steps: {metrics.get('train_steps', 'N/A')}")
    print(f"  Final loss: {metrics.get('train_loss', 'N/A'):.4f}")

    # DPO-specific metrics
    for key in ["rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    print(f"\nTo use this model:")
    print(f"  midas draft --provider local --model {final_dir}")


if __name__ == "__main__":
    main()
