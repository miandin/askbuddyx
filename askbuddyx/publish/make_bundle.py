#!/usr/bin/env python3
"""Create Hugging Face bundle for publishing."""

import argparse
import json
import os
import shutil

from askbuddyx.config import ADAPTER_DIR, BUNDLE_DIR, HOST, PORT, SERVED_MODEL_NAME


def make_bundle(
    adapter_dir: str = ADAPTER_DIR,
    bundle_dir: str = BUNDLE_DIR,
):
    """
    Create Hugging Face bundle.

    Args:
        adapter_dir: Directory containing adapter artifacts
        bundle_dir: Directory to create bundle in
    """
    print("=" * 60)
    print("Creating Hugging Face Bundle")
    print("=" * 60)
    print(f"Adapter directory: {adapter_dir}")
    print(f"Bundle directory: {bundle_dir}")
    print()

    # Check adapter exists
    if not os.path.exists(adapter_dir):
        print(f"Error: Adapter directory not found: {adapter_dir}")
        return False

    # Create bundle directory
    os.makedirs(bundle_dir, exist_ok=True)

    # Copy adapter artifacts
    print("Copying adapter artifacts...")
    for item in os.listdir(adapter_dir):
        src = os.path.join(adapter_dir, item)
        dst = os.path.join(bundle_dir, item)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {item}")
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  Copied directory: {item}")

    print()

    # Copy documentation files
    print("Copying documentation files...")
    doc_files = ["README.md", "MODEL_CARD.md", "LICENSE-THIRD-PARTY.md"]
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            shutil.copy2(doc_file, os.path.join(bundle_dir, doc_file))
            print(f"  Copied: {doc_file}")
        else:
            print(f"  Warning: {doc_file} not found")

    print()

    # Create config.json for HF download tracking
    print("Creating config.json for download tracking...")
    run_meta_path = os.path.join(adapter_dir, "run_meta.json")
    training_iters = 50  # default
    if os.path.exists(run_meta_path):
        with open(run_meta_path, "r") as f:
            meta = json.load(f)
            training_iters = meta.get("iters", 50)
    
    config_json = {
        "model_type": "qwen2",
        "adapter_type": "lora",
        "base_model": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        "base_model_reference": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "task": "text-generation",
        "framework": "mlx",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "trained_on": "flwrlabs/code-alpaca-20k",
        "training_samples": 2000,
        "training_iterations": training_iters,
        "model_name": "AskBuddyX",
        "description": "LoRA adapter for Qwen2.5-Coder-0.5B-Instruct trained on code-alpaca-20k dataset. Provides runnable-first coding assistance.",
        "license": "apache-2.0"
    }
    config_path = os.path.join(bundle_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=2)
    print("  Created: config.json")

    print()

    # Create usage snippet
    print("Creating usage snippet...")
    usage_snippet = f"""# AskBuddyX Usage

## Quick Start

### 1. Install dependencies
```bash
pip install mlx-lm
```

### 2. Start the server
```bash
# Using the base model with this adapter
python -m mlx_lm.server \\
  --model mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit \\
  --adapter-path . \\
  --host {HOST} \\
  --port {PORT}
```

### 3. Test with curl
```bash
curl http://{HOST}:{PORT}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "{SERVED_MODEL_NAME}",
    "messages": [
      {{"role": "user", "content": "Write a Python function to add two numbers"}}
    ],
    "max_tokens": 256
  }}'
```

## Response Format

AskBuddyX provides runnable-first responses with these sections:
- **Solution**: Main implementation
- **Usage**: Smallest runnable example
- **Sanity test**: Tiny test snippet (when appropriate)
"""

    usage_path = os.path.join(bundle_dir, "USAGE.md")
    with open(usage_path, "w") as f:
        f.write(usage_snippet)
    print("  Created: USAGE.md")

    print()
    print("=" * 60)
    print("Bundle created successfully!")
    print("=" * 60)
    print(f"Bundle location: {bundle_dir}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Create Hugging Face bundle")
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=ADAPTER_DIR,
        help=f"Adapter directory (default: {ADAPTER_DIR})",
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default=BUNDLE_DIR,
        help=f"Bundle directory (default: {BUNDLE_DIR})",
    )

    args = parser.parse_args()
    success = make_bundle(
        adapter_dir=args.adapter_dir,
        bundle_dir=args.bundle_dir,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()

