#!/usr/bin/env python3
"""Run LoRA training using MLX."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from askbuddyx.config import (
    ADAPTER_DIR,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MODEL_ID,
    TRAIN_ITERS,
    TRAINING_READY_DIR,
)


def discover_mlx_lora_flags():
    """Discover available MLX LoRA CLI flags."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlx_lm.lora", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout
    except Exception as e:
        print(f"Warning: Could not discover MLX LoRA flags: {e}")
        return ""


def run_lora_training(
    model_id: str = MODEL_ID,
    data_dir: str = TRAINING_READY_DIR,
    output_dir: str = ADAPTER_DIR,
    iters: int = TRAIN_ITERS,
    rank: int = LORA_RANK,
    alpha: int = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    lr: float = LEARNING_RATE,
):
    """
    Run LoRA training using MLX.

    Args:
        model_id: Model ID to use
        data_dir: Directory containing train.jsonl and val.jsonl
        output_dir: Directory to save adapter
        iters: Number of training iterations
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        lr: Learning rate
    """
    print("=" * 60)
    print("Starting LoRA Training")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Iterations: {iters}")
    print(f"LoRA rank: {rank}, alpha: {alpha}, dropout: {dropout}")
    print(f"Learning rate: {lr}")
    print()

    # Discover available flags
    print("Discovering MLX LoRA CLI flags...")
    help_text = discover_mlx_lora_flags()
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build command - MLX LoRA expects --data to point to directory
    # containing train.jsonl and valid.jsonl (not val.jsonl)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model", model_id,
        "--train",
        "--data", data_dir,  # Point to directory, not individual file
        "--iters", str(iters),
        "--adapter-path", output_dir,
    ]

    # Add optional parameters if they appear to be supported
    if "--lora-layers" in help_text or "lora-layers" in help_text:
        cmd.extend(["--lora-layers", str(rank)])

    if "--learning-rate" in help_text or "learning-rate" in help_text:
        cmd.extend(["--learning-rate", str(lr)])

    print("Running command:")
    print(" ".join(cmd))
    print()

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"Training failed with exit code {e.returncode}")
        print("=" * 60)
        sys.exit(1)

    # Write metadata
    metadata = {
        "model_id": model_id,
        "dataset_id": "flwrlabs/code-alpaca-20k",
        "iters": iters,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "learning_rate": lr,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    metadata_path = os.path.join(output_dir, "run_meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LoRA training")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=TRAINING_READY_DIR,
        help=f"Data directory (default: {TRAINING_READY_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=ADAPTER_DIR,
        help=f"Output directory (default: {ADAPTER_DIR})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=TRAIN_ITERS,
        help=f"Training iterations (default: {TRAIN_ITERS})",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=LORA_RANK,
        help=f"LoRA rank (default: {LORA_RANK})",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=LORA_ALPHA,
        help=f"LoRA alpha (default: {LORA_ALPHA})",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=LORA_DROPOUT,
        help=f"LoRA dropout (default: {LORA_DROPOUT})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )

    args = parser.parse_args()
    run_lora_training(
        model_id=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        iters=args.iters,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

