#!/usr/bin/env python3
"""Prepare and clean the dataset for training."""

import argparse
import hashlib
import json
import os
import re

from askbuddyx.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Patterns for secret redaction
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----", re.DOTALL), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"sk-[a-zA-Z0-9]{32,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "[REDACTED_GOOGLE_API_KEY]"),
]


def redact_secrets(text: str) -> str:
    """Redact potential secrets from text."""
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def compute_hash(instruction: str, input_text: str, output: str) -> str:
    """Compute hash for deduplication."""
    content = f"{instruction}|{input_text}|{output}"
    return hashlib.sha256(content.encode()).hexdigest()


def prepare_dataset(
    input_path: str = None,
    output_dir: str = PROCESSED_DATA_DIR,
    train_split: float = 0.98,
):
    """
    Prepare dataset: redact secrets, deduplicate, and split train/val.

    Args:
        input_path: Path to raw JSONL file
        output_dir: Directory to save processed data
        train_split: Fraction of data for training (default: 0.98)
    """
    if input_path is None:
        input_path = os.path.join(RAW_DATA_DIR, "codealpaca.jsonl")

    print(f"Reading from: {input_path}")

    # Read and process
    examples = []
    seen_hashes = set()
    redacted_count = 0
    duplicate_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)

            # Redact secrets
            original_output = example["output"]
            example["instruction"] = redact_secrets(example["instruction"])
            example["input"] = redact_secrets(example["input"])
            example["output"] = redact_secrets(example["output"])

            if original_output != example["output"]:
                redacted_count += 1

            # Deduplicate
            example_hash = compute_hash(
                example["instruction"], example["input"], example["output"]
            )
            if example_hash in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes.add(example_hash)
            examples.append(example)

    print(f"Total examples read: {len(examples) + duplicate_count}")
    print(f"Redacted secrets in: {redacted_count} examples")
    print(f"Removed duplicates: {duplicate_count}")
    print(f"Unique examples: {len(examples)}")

    # Split train/val
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Write train
    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Wrote train data to: {train_path}")

    # Write val
    val_path = os.path.join(output_dir, "val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Wrote val data to: {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file (default: data/raw/codealpaca.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f"Output directory (default: {PROCESSED_DATA_DIR})",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.98,
        help="Fraction of data for training (default: 0.98)",
    )

    args = parser.parse_args()
    prepare_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        train_split=args.train_split,
    )


if __name__ == "__main__":
    main()

