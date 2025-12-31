# Training AskBuddyX with Perfect Examples

## Overview

This guide explains how to retrain AskBuddyX using the 20 perfect examples that demonstrate the desired output format with proper docstrings, Solution/Usage/Sanity test sections.

## What Was Created

### Perfect Examples Dataset
**Location**: `data/custom/perfect_examples.jsonl`  
**Count**: 20 examples  
**Format**: Each example follows the exact AskBuddyX format

**Example structure**:
```json
{
  "instruction": "Write a Python function to calculate factorial",
  "input": "",
  "output": "### Solution\n\n```python\ndef factorial(n):\n    \"\"\"\n    Calculate the factorial of a number.\n    \n    Args:\n        n (int): The number to calculate factorial for\n        \n    Returns:\n        int: The factorial of n\n    \"\"\"\n    # implementation\n```\n\n### Usage\n\n```python\nresult = factorial(5)\nprint(result)  # Output: 120\n```\n\n### Sanity test\n\n```python\nassert factorial(0) == 1\nassert factorial(5) == 120\n```"
}
```

### Topics Covered (20 examples)
1. Factorial calculation
2. Fibonacci sequence
3. Prime number checking
4. String reversal
5. Finding maximum in list
6. Palindrome checking
7. List sum calculation
8. Vowel counting
9. Duplicate removal
10. Merging sorted lists
11. GCD calculation
12. Temperature conversion
13. List flattening
14. Power calculation
15. Longest word length
16. Sorted list checking
17. List rotation
18. List intersection
19. Average calculation
20. Finding all primes (Sieve)

---

## Training Options

### Option 1: Perfect Examples Only (Quick Test)

Train using ONLY the 20 perfect examples to see if the model learns the format.

**Pros**:
- Fast training (20 examples)
- Pure format learning
- Quick validation

**Cons**:
- Limited knowledge
- May overfit to these specific tasks
- Less robust

**Steps**:
```bash
# 1. Modify fetch script to use custom data
export CUSTOM_DATA_ONLY=true

# 2. Train with more iterations (need more to learn from fewer examples)
export TRAIN_ITERS=200

# 3. Run training
make train

# 4. Test
python askbuddyx/eval/run_sanity_prompts.py --adapter outputs/adapters/dev
```

### Option 2: Mixed Training (Recommended)

Combine perfect examples with code-alpaca for both format AND knowledge.

**Pros**:
- Learns proper format from perfect examples
- Gains broad knowledge from code-alpaca
- Best balance

**Cons**:
- Slightly longer training
- Need to merge datasets

**Steps**:
```bash
# 1. Merge datasets (30% perfect, 70% code-alpaca)
python scripts/merge_datasets.py \
  --datasets data/custom/perfect_examples.jsonl flwrlabs/code-alpaca-20k \
  --weights 0.3 0.7 \
  --output data/merged/training_data.jsonl \
  --limit 2000

# 2. Update fetch script to use merged data
# (See "Modifying Fetch Script" section below)

# 3. Train with more iterations
export TRAIN_ITERS=500

# 4. Run full pipeline
make all

# 5. Publish
make publish
```

### Option 3: Incremental Training

Start with perfect examples, then add code-alpaca knowledge.

**Pros**:
- Learns format first
- Then expands knowledge
- Most controlled approach

**Cons**:
- Requires two training runs
- More time consuming

**Steps**:
```bash
# Phase 1: Learn format (perfect examples only)
export TRAIN_ITERS=200
# Modify fetch to use perfect_examples.jsonl
make train

# Phase 2: Expand knowledge (add code-alpaca)
export TRAIN_ITERS=500
# Modify fetch to use merged data
make train

# Publish
make publish
```

---

## Implementation Guide

### Step 1: Merge Datasets (Recommended Approach)

```bash
# Create merged dataset with 30% perfect examples, 70% code-alpaca
python scripts/merge_datasets.py \
  --datasets data/custom/perfect_examples.jsonl flwrlabs/code-alpaca-20k \
  --weights 0.3 0.7 \
  --output data/merged/training_data.jsonl \
  --limit 2000 \
  --shuffle
```

**Result**: `data/merged/training_data.jsonl` with:
- ~600 perfect examples (30% of 2000)
- ~1400 code-alpaca examples (70% of 2000)
- Shuffled for better learning

### Step 2: Modify Fetch Script

Edit `askbuddyx/train/fetch_codealpaca.py` to use merged data:

```python
# Option A: Load from local merged file
def fetch_dataset(limit=DATA_LIMIT):
    """Fetch dataset from local merged file."""
    print("Loading merged dataset...")
    
    # Load from local JSONL
    dataset = load_dataset(
        "json",
        data_files="data/merged/training_data.jsonl",
        split="train"
    )
    
    # Rest of the function remains the same
    ...
```

OR create a new fetch script:

```bash
# Create new fetch script for merged data
cp askbuddyx/train/fetch_codealpaca.py askbuddyx/train/fetch_merged.py
```

Then edit `fetch_merged.py` to load from `data/merged/training_data.jsonl`.

### Step 3: Update Makefile (Optional)

Add a new target for merged training:

```makefile
# Add to Makefile
train-merged:
	@echo "Training with merged dataset..."
	python -m askbuddyx.train.fetch_merged
	python -m askbuddyx.train.prepare_dataset
	python -m askbuddyx.train.build_training_text
	python -m askbuddyx.train.run_lora
	@echo "✓ Training complete"
```

### Step 4: Configure Training Parameters

Edit `askbuddyx/config.py` or use environment variables:

```bash
# Increase iterations for better learning
export TRAIN_ITERS=500

# Optionally increase data limit
export DATA_LIMIT=2000
```

### Step 5: Run Training

```bash
# Full pipeline with merged data
make all

# Or step by step
make fetch-data    # Fetches merged data
make prep-data     # Preprocesses
make train         # Trains with 500 iterations
make eval          # Evaluates
```

### Step 6: Test the Results

```bash
# Test with sanity prompts
python askbuddyx/eval/run_sanity_prompts.py --adapter outputs/adapters/dev

# Test with code generation
python askbuddyx/eval/run_codegen_smoke.py --adapter outputs/adapters/dev

# Test with custom prompt
python scripts/test_hf_inference.py \
  --repo salakash/AskBuddyX \
  --prompt "Write a Python function to calculate factorial"
```

### Step 7: Publish Updated Model

```bash
# Create bundle and publish
make publish
```

---

## Expected Results

### Before (Current Model - 50 iterations, code-alpaca only)
```python
# Prompt: "Write a Python function to calculate factorial"

# Output:
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
```

❌ No docstring  
❌ No Args/Returns  
❌ No ### Solution section  
❌ No ### Usage example  
❌ No ### Sanity test

### After (New Model - 500 iterations, 30% perfect + 70% code-alpaca)
```python
# Prompt: "Write a Python function to calculate factorial"

# Output:
### Solution

```python
def factorial(n):
    """
    Calculate the factorial of a number.
    
    Args:
        n (int): The number to calculate factorial for
        
    Returns:
        int: The factorial of n
    """
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
```

### Usage

```python
result = factorial(5)
print(result)  # Output: 120
```

### Sanity test

```python
assert factorial(0) == 1
assert factorial(5) == 120
assert factorial(1) == 1
```
```

✅ Complete docstring  
✅ Args and Returns documented  
✅ ### Solution section  
✅ ### Usage example  
✅ ### Sanity test included

---

## Training Time Estimates

### Hardware: M1 Mac, 32GB RAM

**Perfect Examples Only (20 examples)**:
- 200 iterations: ~5-10 minutes
- 500 iterations: ~10-20 minutes

**Merged Dataset (2000 examples)**:
- 200 iterations: ~30-45 minutes
- 500 iterations: ~60-90 minutes
- 1000 iterations: ~2-3 hours

**Factors affecting time**:
- Model size (0.5B is fast)
- Batch size (default: 4)
- Sequence length (default: 2048)
- MLX optimization (Apple Silicon)

---

## Troubleshooting

### Issue 1: Model Still Not Using Format

**Possible causes**:
1. Not enough iterations (try 500-1000)
2. Perfect examples ratio too low (increase to 50%)
3. Data not properly merged

**Solution**:
```bash
# Increase perfect examples ratio
python scripts/merge_datasets.py \
  --datasets data/custom/perfect_examples.jsonl flwrlabs/code-alpaca-20k \
  --weights 0.5 0.5 \
  --output data/merged/training_data.jsonl \
  --limit 2000

# Train with more iterations
export TRAIN_ITERS=1000
make train
```

### Issue 2: Model Overfits to Perfect Examples

**Symptoms**: Only works for the 20 specific tasks

**Solution**:
```bash
# Reduce perfect examples ratio
python scripts/merge_datasets.py \
  --datasets data/custom/perfect_examples.jsonl flwrlabs/code-alpaca-20k \
  --weights 0.2 0.8 \
  --output data/merged/training_data.jsonl \
  --limit 2000
```

### Issue 3: Training Takes Too Long

**Solution**:
```bash
# Reduce data limit
export DATA_LIMIT=1000

# Or reduce iterations
export TRAIN_ITERS=300

# Or use smaller batch size
# Edit askbuddyx/train/run_lora.py: batch_size=2
```

### Issue 4: Out of Memory

**Solution**:
```bash
# Reduce batch size in run_lora.py
# Change: batch_size=4 to batch_size=2

# Or reduce max sequence length
# Change: max_seq_length=2048 to max_seq_length=1024
```

---

## Monitoring Training

### Check Training Progress

```bash
# Watch training output
tail -f outputs/adapters/dev/training.log

# Check adapter files
ls -lh outputs/adapters/dev/

# View training metadata
cat outputs/adapters/dev/run_meta.json
```

### Evaluate During Training

```bash
# Run evaluation after training
python askbuddyx/eval/run_sanity_prompts.py --adapter outputs/adapters/dev

# Check specific prompt
python -c "
from mlx_lm import load, generate
model, tokenizer = load('mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit', 
                        adapter_path='outputs/adapters/dev')
response = generate(model, tokenizer, 
                   prompt='Write a Python function to calculate factorial',
                   max_tokens=500)
print(response)
"
```

---

## Next Steps After Training

### 1. Evaluate Quality
```bash
# Run all evaluations
make eval

# Test with various prompts
python scripts/test_hf_inference.py --interactive
```

### 2. Compare with Previous Version
```bash
# Test old version (from HF)
python scripts/test_hf_inference.py \
  --repo salakash/AskBuddyX \
  --prompt "Write a Python function to calculate factorial"

# Test new version (local)
python askbuddyx/eval/run_sanity_prompts.py \
  --adapter outputs/adapters/dev
```

### 3. Publish if Satisfied
```bash
# Create bundle
make bundle

# Publish to HF
make publish
```

### 4. Document Changes
Update `MODEL_CARD.md` with:
- New training details
- Improved output format
- Example outputs
- Training data composition

---

## Advanced: Creating More Perfect Examples

If you want to add more perfect examples beyond the initial 20:

### Template for New Examples

```json
{
  "instruction": "[Clear task description]",
  "input": "",
  "output": "### Solution\n\n```python\ndef function_name(params):\n    \"\"\"\n    [Brief description]\n    \n    Args:\n        param1 (type): Description\n        \n    Returns:\n        type: Description\n    \"\"\"\n    # Implementation\n    pass\n```\n\n### Usage\n\n```python\nresult = function_name(example_input)\nprint(result)  # Output: expected_output\n```\n\n### Sanity test\n\n```python\nassert function_name(test1) == expected1\nassert function_name(test2) == expected2\n```"
}
```

### Add to Existing File

```bash
# Append new examples to perfect_examples.jsonl
echo '{"instruction": "...", "input": "", "output": "..."}' >> data/custom/perfect_examples.jsonl

# Verify format
python -c "
import json
with open('data/custom/perfect_examples.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except:
            print(f'Error on line {i}')
"
```

### Suggested Additional Topics

- Binary search
- Quicksort/Mergesort
- Tree traversal
- Graph algorithms
- String manipulation
- Data structure implementations
- File I/O operations
- Error handling patterns
- Class definitions
- Decorator examples

---

## Summary

✅ **Created**: 20 perfect training examples in `data/custom/perfect_examples.jsonl`  
✅ **Format**: Each example demonstrates proper docstrings and AskBuddyX structure  
✅ **Recommended**: Mix 30% perfect + 70% code-alpaca with 500 iterations  
✅ **Expected**: Model will learn to consistently use the desired format  
✅ **Timeline**: ~60-90 minutes training on M1 Mac

**Next Action**: Run the merge and training commands above to create an improved model!