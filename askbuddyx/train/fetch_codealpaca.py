#!/usr/bin/env python3
"""Fetch and normalize the code-alpaca-20k dataset."""

import argparse
import json
import os

from datasets import load_dataset

from askbuddyx.config import DATA_LIMIT, DATASET_ID, RAW_DATA_DIR


def fetch_dataset(limit: int = DATA_LIMIT, output_dir: str = RAW_DATA_DIR):
    """
    Fetch code-alpaca-20k dataset and save as JSONL.

    Args:
        limit: Maximum number of examples to fetch
        output_dir: Directory to save raw data
    """
    print(f"Fetching dataset: {DATASET_ID}")
    print(f"Limit: {limit}")

    # Load dataset
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Total examples in dataset: {len(dataset)}")

    # Limit if specified
    if limit and limit < len(dataset):
        dataset = dataset.select(range(limit))
        print(f"Limited to: {len(dataset)} examples")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "codealpaca.jsonl")

    # Normalize and write JSONL
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            # Normalize field names
            normalized = {
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "output": example.get("output", ""),
            }
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} examples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fetch code-alpaca-20k dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=DATA_LIMIT,
        help=f"Maximum number of examples to fetch (default: {DATA_LIMIT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RAW_DATA_DIR,
        help=f"Output directory (default: {RAW_DATA_DIR})",
    )

    args = parser.parse_args()
    fetch_dataset(limit=args.limit, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

