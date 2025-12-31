#!/usr/bin/env python3
"""Build training-ready text from processed dataset."""

import argparse
import json
import os

from askbuddyx.config import PROCESSED_DATA_DIR, TRAINING_READY_DIR
from askbuddyx.prompting import build_training_prompt

MAX_INSTRUCTION_LENGTH = 1024
MAX_INPUT_LENGTH = 512


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length characters."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def build_training_text(
    input_dir: str = PROCESSED_DATA_DIR,
    output_dir: str = TRAINING_READY_DIR,
):
    """
    Build training-ready text files from processed dataset.

    Args:
        input_dir: Directory containing processed train.jsonl and val.jsonl
        output_dir: Directory to save training-ready files
    """
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val"]:
        input_path = os.path.join(input_dir, f"{split}.jsonl")
        # MLX LoRA expects "valid.jsonl" not "val.jsonl"
        output_split = "valid" if split == "val" else split
        output_path = os.path.join(output_dir, f"{output_split}.jsonl")

        print(f"Processing {split} split...")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")

        count = 0
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                example = json.loads(line)

                # Truncate long inputs for faster training
                instruction = truncate_text(
                    example["instruction"], MAX_INSTRUCTION_LENGTH
                )
                input_text = truncate_text(
                    example["input"], MAX_INPUT_LENGTH
                )
                output = example["output"]

                # Build training prompt
                text = build_training_prompt(instruction, input_text, output)

                # Write as JSONL with "text" field
                training_example = {"text": text}
                fout.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                count += 1

        print(f"  Wrote {count} examples")


def main():
    parser = argparse.ArgumentParser(description="Build training-ready text")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f"Input directory (default: {PROCESSED_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=TRAINING_READY_DIR,
        help=f"Output directory (default: {TRAINING_READY_DIR})",
    )

    args = parser.parse_args()
    build_training_text(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

