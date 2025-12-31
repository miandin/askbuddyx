#!/usr/bin/env python3
"""
Merge multiple datasets for training
"""
import json
import argparse
from pathlib import Path

def merge_datasets(
    base_file="data/raw/codealpaca.jsonl",
    custom_file="data/custom/my_examples.jsonl",
    output_file="data/raw/merged.jsonl"
):
    """Merge base dataset with custom examples"""
    
    merged = []
    
    # Load base dataset
    if Path(base_file).exists():
        with open(base_file) as f:
            for line in f:
                merged.append(json.loads(line))
        print(f"Loaded {len(merged)} examples from base dataset: {base_file}")
    else:
        print(f"Warning: Base file {base_file} not found")
    
    # Load custom dataset
    custom_count = 0
    if Path(custom_file).exists():
        with open(custom_file) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    merged.append(json.loads(line))
                    custom_count += 1
        print(f"Added {custom_count} custom examples from: {custom_file}")
    else:
        print(f"Warning: Custom file {custom_file} not found")
    
    # Write merged dataset
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for item in merged:
            f.write(json.dumps(item) + '\n')
    
    print(f"Total: {len(merged)} examples written to {output_file}")
    return len(merged)

def merge_multiple_datasets(
    datasets=[
        "data/raw/codealpaca.jsonl",
        "data/raw/mbpp.jsonl",
        "data/raw/humaneval.jsonl",
        "data/custom/my_examples.jsonl"
    ],
    output_file="data/raw/multi_merged.jsonl",
    weights=None
):
    """Merge multiple datasets with optional weighting"""
    
    merged = []
    
    for i, dataset_file in enumerate(datasets):
        if not Path(dataset_file).exists():
            print(f"Warning: {dataset_file} not found, skipping")
            continue
        
        with open(dataset_file) as f:
            examples = [json.loads(line) for line in f]
        
        # Apply weighting if specified
        if weights and i < len(weights):
            target_count = int(len(examples) * weights[i])
            examples = examples[:target_count]
        
        merged.extend(examples)
        print(f"Added {len(examples)} from {dataset_file}")
    
    # Shuffle for better training
    import random
    random.shuffle(merged)
    
    # Write merged dataset
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for item in merged:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nTotal: {len(merged)} examples in {output_file}")
    return len(merged)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge datasets for training")
    parser.add_argument("--base", default="data/raw/codealpaca.jsonl", help="Base dataset file")
    parser.add_argument("--custom", default="data/custom/my_examples.jsonl", help="Custom dataset file")
    parser.add_argument("--output", default="data/raw/merged.jsonl", help="Output file")
    parser.add_argument("--multi", action="store_true", help="Merge multiple datasets")
    
    args = parser.parse_args()
    
    if args.multi:
        merge_multiple_datasets()
    else:
        merge_datasets(args.base, args.custom, args.output)
