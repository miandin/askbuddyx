#!/usr/bin/env python3
"""Duplicate perfect examples to increase their ratio in training data."""

import json
import sys

def duplicate_examples(input_file, output_file, times=10):
    """
    Duplicate each example in the input file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        times: Number of times to duplicate each example
    """
    examples = []
    
    # Read all examples
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    print(f"Read {len(examples)} examples from {input_file}")
    
    # Duplicate each example
    duplicated = []
    for example in examples:
        for _ in range(times):
            duplicated.append(example)
    
    # Write duplicated examples
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in duplicated:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Wrote {len(duplicated)} examples to {output_file}")
    print(f"Duplication factor: {times}x")
    print(f"Original: {len(examples)} → Duplicated: {len(duplicated)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python duplicate_perfect_examples.py <input_file> <output_file> [times]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    times = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    duplicate_examples(input_file, output_file, times)
