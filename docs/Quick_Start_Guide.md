# AskBuddyX Quick Start Guide

**Complete workflow from base model to Hugging Face in simple steps**

---

## Prerequisites

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install mlx-lm datasets pyyaml huggingface_hub ruff pytest flask

# 3. Login to Hugging Face (one-time)
huggingface-cli login
```

---

## Step 1: Choose Your Components

### Base Model
```python
# In askbuddyx/config.py
MODEL_ID = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
```
**Why**: MLX-optimized, 4-bit quantized, runs on Apple Silicon

### Dataset
```python
# In askbuddyx/config.py
DATASET_ID = "flwrlabs/code-alpaca-20k"
DATA_LIMIT = 1000  # Start small
```
**Why**: Apache-2.0 license, coding-focused

### Training Parameters
```python
# In askbuddyx/config.py
TRAIN_ITERS = 100      # Iterations
DATA_LIMIT = 1000      # Dataset size
```

---

## Step 2: Train Your Adapter

### One Command Training
```bash
make all
```

**What it does**:
1. Fetches dataset (1000 examples)
2. Preprocesses (removes secrets, deduplicates)
3. Builds training text
4. Trains LoRA adapter (100 iterations)
5. Evaluates output

**Time**: ~2-3 minutes on M1 Mac

---

## Step 3: Test Your Model

### Quick Test
```bash
python scripts/test_output_format.py
```

**Checks**:
- ✅ Has `### Solution` section
- ✅ Has `### Usage` section
- ✅ Has `### Sanity test` section
- ✅ Has proper docstrings

### Demo Multiple Prompts
```bash
python scripts/demo_prompts.py
```

Tests 5 different coding prompts

---

## Step 4: Publish to Hugging Face

### One Command Publish
```bash
make publish
```

**What it does**:
1. Creates bundle with adapter + docs
2. Uploads to `salakash/AskBuddyX`
3. Includes MODEL_CARD.md and LICENSE

**Result**: https://huggingface.co/salakash/AskBuddyX

---

## Step 5: Iterate and Improve

### Option A: Adjust Training Parameters

```bash
# Train longer
TRAIN_ITERS=200 make train

# Use more data
DATA_LIMIT=2000 make all
```

### Option B: Add Custom Examples

```bash
# 1. Create custom examples
echo '{"instruction":"Your task","input":"","output":"Your solution"}' >> data/custom/my_examples.jsonl

# 2. Merge with dataset
python scripts/merge_datasets.py \
  --base data/custom/my_examples.jsonl \
  --custom data/raw/codealpaca.jsonl \
  --output data/raw/codealpaca.jsonl

# 3. Retrain
make prep-data train
```

### Option C: Change Base Model

```python
# Edit askbuddyx/config.py
MODEL_ID = "mlx-community/different-model-4bit"

# Retrain
make all
```

---

## Step 6: Re-publish Updates

```bash
# After any changes
make train      # Train with new settings
make publish    # Publish updated adapter
```

**Note**: Each publish overwrites previous version on Hugging Face

---

## Quick Reference

### Essential Commands

| Command | Purpose |
|---------|---------|
| `make all` | Complete training pipeline |
| `make train` | Train adapter only |
| `make publish` | Publish to Hugging Face |
| `make serve` | Start OpenAI-compatible server |
| `python webui/app.py` | Start Web UI |

### Key Files

| File | Purpose |
|------|---------|
| `askbuddyx/config.py` | All configuration |
| `data/custom/*.jsonl` | Your custom examples |
| `outputs/adapters/dev/` | Trained adapter |
| `outputs/hf_bundle/` | Ready to publish |

### Configuration Tweaks

```python
# askbuddyx/config.py

# Training
TRAIN_ITERS = 100        # More = better learning
DATA_LIMIT = 1000        # More = more diverse

# LoRA (advanced)
LORA_RANK = 8           # Higher = more capacity
LORA_ALPHA = 16         # Scaling factor
LORA_DROPOUT = 0.05     # Regularization
```

---

## Troubleshooting

### Model not improving?
```bash
# Increase iterations
TRAIN_ITERS=200 make train

# Add more custom examples (15-20% of dataset)
# See "Option B" above
```

### Out of memory?
```bash
# Reduce batch size or use smaller model
# MLX handles this automatically on M1
```

### Wrong output format?
```bash
# Add more examples showing desired format
# Aim for 15-20% of training data
```

---

## Success Metrics

**Good Training**:
- Loss drops from ~3.0 to <0.5
- Validation loss similar to training loss
- Test script shows ✅ for all checks

**Good Model**:
- 60%+ prompts use structured format
- Proper docstrings with Args/Returns
- Runnable examples in Usage section
- Sanity tests with assertions

---

## Next Steps

1. **Experiment**: Try different TRAIN_ITERS (50, 100, 200)
2. **Customize**: Add 50-100 examples of your desired output
3. **Evaluate**: Use `scripts/demo_prompts.py` to test
4. **Publish**: Share on Hugging Face with `make publish`
5. **Iterate**: Repeat based on results

**Remember**: Start small (100 iterations, 1000 examples), test, then scale up!