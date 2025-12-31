# AskBuddyX Iterative Fine-Tuning Guide

**Version**: 1.0  
**Last Updated**: 2025-12-29  
**Purpose**: Complete guide for continuous model improvement through iterative fine-tuning

---

## Table of Contents

1. [Overview](#overview)
2. [Best Practices for Iterative Fine-Tuning](#best-practices)
3. [Approach 1: More Training Iterations](#approach-1-more-iterations)
4. [Approach 2: Adding Custom JSONL Data](#approach-2-custom-data)
5. [Approach 3: Incorporating Additional Datasets](#approach-3-additional-datasets)
6. [Approach 4: Incremental Fine-Tuning](#approach-4-incremental)
7. [Version Management Strategy](#version-management)
8. [Publishing Updates to Hugging Face](#publishing-updates)
9. [Evaluation & Quality Control](#evaluation)
10. [Production Workflow](#production-workflow)
11. [Troubleshooting](#troubleshooting)

---

## 1. Overview {#overview}

AskBuddyX uses **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning. This guide covers four main approaches for iterative improvement:

| Approach | Use Case | Effort | Impact |
|----------|----------|--------|--------|
| **More Iterations** | Quick improvement on existing data | Low | Medium |
| **Custom JSONL** | Domain-specific knowledge injection | Medium | High |
| **Additional Datasets** | Broader capabilities | High | High |
| **Incremental Fine-Tuning** | Continuous learning from production | Medium | Very High |

### Current Baseline
- **Model**: Qwen2.5-Coder-0.5B-Instruct (4-bit MLX)
- **Dataset**: code-alpaca-20k (2,000 examples)
- **Training**: 50 iterations, rank=8, alpha=16
- **Results**: Training loss 0.298, Validation loss 0.357
- **Adapter Size**: 11.8MB (2.933M parameters)

---

## 2. Best Practices for Iterative Fine-Tuning {#best-practices}

### 2.1 General Principles

1. **Version Everything**: Track model versions, datasets, hyperparameters
2. **Evaluate Before Publishing**: Always run sanity checks and codegen tests
3. **Keep Adapters Separate**: Don't overwrite previous versions immediately
4. **Monitor Metrics**: Track loss curves, eval scores, inference quality
5. **Document Changes**: Update MODEL_CARD.md with each iteration
6. **Test in Staging**: Use dev adapters before promoting to production

### 2.2 Quality Metrics to Track

```python
# Key metrics for each training run
metrics = {
    "training_loss": 0.298,           # Lower is better
    "validation_loss": 0.357,         # Watch for overfitting
    "sanity_prompts_passed": "10/10", # Must be 100%
    "codegen_tests_passed": "2/2",    # Must be 100%
    "avg_response_time": "2.3s",      # Inference speed
    "response_quality": "manual",     # Human evaluation
}
```

### 2.3 When to Stop Training

**Stop if:**
- Validation loss starts increasing (overfitting)
- Sanity prompts fail
- Generated code becomes less coherent
- Model starts hallucinating APIs

**Continue if:**
- Both losses decreasing
- Eval scores improving
- Response quality improving

---

## 3. Approach 1: More Training Iterations {#approach-1-more-iterations}

**Best for**: Quick improvement without changing data

### 3.1 Current Setup
```bash
# Current: 50 iterations (dev/testing)
TRAIN_ITERS=50 make train
```

### 3.2 Recommended Iteration Ranges

| Stage | Iterations | Purpose | Expected Time (M1) |
|-------|-----------|---------|-------------------|
| **Dev** | 50-100 | Quick testing | 5-10 min |
| **Staging** | 200-500 | Quality check | 20-50 min |
| **Production** | 1000-2000 | Full training | 2-4 hours |
| **Extended** | 3000-5000 | Maximum quality | 6-10 hours |

### 3.3 Step-by-Step: Increase Iterations

#### Step 1: Set Iteration Count
```bash
# Option A: Environment variable
export TRAIN_ITERS=1000

# Option B: Direct in config
# Edit askbuddyx/config.py
TRAIN_ITERS = 1000
```

#### Step 2: Create Versioned Output
```bash
# Create version-specific output directory
mkdir -p outputs/adapters/v1.1

# Modify train/run_lora.py to use versioned output
# Or use environment variable
export OUTPUT_DIR="outputs/adapters/v1.1"
```

#### Step 3: Train with More Iterations
```bash
# Full command
TRAIN_ITERS=1000 OUTPUT_DIR="outputs/adapters/v1.1" make train

# Or modify Makefile train target:
# train:
#     python -m askbuddyx.train.run_lora --iters $(TRAIN_ITERS) --output $(OUTPUT_DIR)
```

#### Step 4: Evaluate New Adapter
```bash
# Test with new adapter
python -m askbuddyx.eval.run_sanity_prompts \
    --adapter outputs/adapters/v1.1

python -m askbuddyx.eval.run_codegen_smoke \
    --adapter outputs/adapters/v1.1
```

#### Step 5: Compare Results
```bash
# Compare loss curves
cat outputs/adapters/dev/run_meta.json
cat outputs/adapters/v1.1/run_meta.json

# Look for:
# - Lower final losses
# - Smooth convergence
# - No overfitting (val_loss < train_loss + 0.1)
```

### 3.4 Monitoring Training Progress

Create a simple monitoring script:

```python
# scripts/monitor_training.py
import json
import time
from pathlib import Path

def monitor_training(output_dir="outputs/adapters/v1.1"):
    meta_file = Path(output_dir) / "run_meta.json"
    
    while True:
        if meta_file.exists():
            with open(meta_file) as f:
                data = json.load(f)
            
            print(f"\n=== Training Progress ===")
            print(f"Iterations: {data.get('iters', 'N/A')}")
            print(f"Training Loss: {data.get('final_train_loss', 'N/A')}")
            print(f"Validation Loss: {data.get('final_val_loss', 'N/A')}")
            print(f"Status: {data.get('status', 'Running')}")
            
            if data.get('status') == 'completed':
                break
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    monitor_training()
```

---

## 4. Approach 2: Adding Custom JSONL Data {#approach-2-custom-data}

**Best for**: Domain-specific knowledge, fixing specific weaknesses

### 4.1 JSONL Format

Your custom data must follow this format:

```jsonl
{"instruction": "Write a function to calculate fibonacci", "input": "", "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"}
{"instruction": "Create a REST API endpoint", "input": "Use Flask", "output": "from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('/api/data')\ndef get_data():\n    return jsonify({'status': 'success'})"}
```

### 4.2 Step-by-Step: Inject Custom Data

#### Step 1: Create Custom JSONL File
```bash
# Create directory
mkdir -p data/custom

# Create your custom data file
cat > data/custom/my_examples.jsonl << 'EOF'
{"instruction": "Explain async/await in Python", "input": "", "output": "async/await in Python enables asynchronous programming...\n\n### Solution\n```python\nimport asyncio\n\nasync def fetch_data():\n    await asyncio.sleep(1)\n    return 'data'\n```\n\n### Usage\n```python\nresult = asyncio.run(fetch_data())\nprint(result)\n```"}
{"instruction": "Create a database connection pool", "input": "Use SQLAlchemy", "output": "### Solution\n```python\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.pool import QueuePool\n\nengine = create_engine(\n    'postgresql://user:pass@localhost/db',\n    poolclass=QueuePool,\n    pool_size=5,\n    max_overflow=10\n)\n```\n\n### Usage\n```python\nwith engine.connect() as conn:\n    result = conn.execute('SELECT 1')\n```"}
EOF
```

#### Step 2: Merge with Existing Data

Create a merge script:

```python
# scripts/merge_datasets.py
import json
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
        print(f"Loaded {len(merged)} examples from base dataset")
    
    # Load custom dataset
    custom_count = 0
    if Path(custom_file).exists():
        with open(custom_file) as f:
            for line in f:
                merged.append(json.loads(line))
                custom_count += 1
        print(f"Added {custom_count} custom examples")
    
    # Write merged dataset
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for item in merged:
            f.write(json.dumps(item) + '\n')
    
    print(f"Total: {len(merged)} examples written to {output_file}")
    return len(merged)

if __name__ == "__main__":
    merge_datasets()
```

#### Step 3: Update Training Pipeline

```bash
# Run merge script
python scripts/merge_datasets.py

# Update prepare_dataset.py to use merged data
# Or modify fetch_codealpaca.py to include custom data

# Option A: Modify config to point to merged file
export RAW_DATA_FILE="data/raw/merged.jsonl"

# Option B: Update prepare_dataset.py
# Change input file from codealpaca.jsonl to merged.jsonl
```

#### Step 4: Train with Merged Data

```bash
# Prepare merged dataset
python -m askbuddyx.train.prepare_dataset \
    --input data/raw/merged.jsonl \
    --output data/processed/

# Build training text
python -m askbuddyx.train.build_training_text

# Train new version
TRAIN_ITERS=1000 OUTPUT_DIR="outputs/adapters/v1.2-custom" make train
```

#### Step 5: Validate Custom Knowledge

Create custom eval prompts:

```python
# scripts/test_custom_knowledge.py
from mlx_lm import load, generate

# Load model with new adapter
model, tokenizer = load(
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    adapter_path="outputs/adapters/v1.2-custom"
)

# Test custom knowledge
test_prompts = [
    "Explain async/await in Python",
    "Create a database connection pool using SQLAlchemy",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=500,
        verbose=False
    )
    print(response)
```

### 4.3 Best Practices for Custom Data

1. **Quality over Quantity**: 100 high-quality examples > 1000 poor ones
2. **Follow AskBuddyX Format**: Use Solution/Usage/Sanity test structure
3. **Diverse Examples**: Cover different scenarios, edge cases
4. **Validate Syntax**: Ensure all code examples are runnable
5. **Balance Dataset**: Don't overwhelm with custom data (10-20% max)

---

## 5. Approach 3: Incorporating Additional Datasets {#approach-3-additional-datasets}

**Best for**: Expanding capabilities, adding new domains

### 5.1 Recommended Datasets

| Dataset | Size | License | Use Case |
|---------|------|---------|----------|
| **code-alpaca-20k** | 20k | Apache-2.0 | General coding (current) |
| **CodeSearchNet** | 2M | MIT | Multi-language code search |
| **APPS** | 10k | MIT | Competitive programming |
| **HumanEval** | 164 | MIT | Python function synthesis |
| **MBPP** | 974 | CC-BY-4.0 | Python programming problems |
| **CodeContests** | 13k | Apache-2.0 | Algorithm problems |
| **CommitPackFT** | 4GB | Apache-2.0 | Code commits |

### 5.2 Step-by-Step: Add New Dataset

#### Step 1: Create Dataset Fetcher

```python
# askbuddyx/train/fetch_additional_dataset.py
from datasets import load_dataset
import json
from pathlib import Path
from askbuddyx.config import DATA_DIR

def fetch_mbpp(limit=None):
    """Fetch MBPP dataset for Python programming problems"""
    
    print("Fetching MBPP dataset...")
    dataset = load_dataset("mbpp", "sanitized", split="train")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    # Convert to AskBuddyX format
    output_file = Path(DATA_DIR) / "raw" / "mbpp.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in dataset:
            # MBPP format: text (description), code (solution), test_list
            record = {
                "instruction": item["text"],
                "input": "",
                "output": item["code"]
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"Wrote {len(dataset)} examples to {output_file}")
    return output_file

def fetch_humaneval(limit=None):
    """Fetch HumanEval dataset"""
    
    print("Fetching HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    output_file = Path(DATA_DIR) / "raw" / "humaneval.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in dataset:
            record = {
                "instruction": item["prompt"],
                "input": "",
                "output": item["canonical_solution"]
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"Wrote {len(dataset)} examples to {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    if args.dataset == "mbpp":
        fetch_mbpp(args.limit)
    elif args.dataset == "humaneval":
        fetch_humaneval(args.limit)
```

#### Step 2: Merge Multiple Datasets

```python
# scripts/merge_multiple_datasets.py
import json
from pathlib import Path

def merge_multiple_datasets(
    datasets=[
        "data/raw/codealpaca.jsonl",
        "data/raw/mbpp.jsonl",
        "data/raw/humaneval.jsonl",
        "data/custom/my_examples.jsonl"
    ],
    output_file="data/raw/multi_merged.jsonl",
    weights=None  # Optional: [0.5, 0.2, 0.2, 0.1] for sampling
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
    merge_multiple_datasets()
```

#### Step 3: Train with Multi-Dataset

```bash
# Fetch additional datasets
python -m askbuddyx.train.fetch_additional_dataset --dataset mbpp --limit 500
python -m askbuddyx.train.fetch_additional_dataset --dataset humaneval

# Merge all datasets
python scripts/merge_multiple_datasets.py

# Prepare and train
python -m askbuddyx.train.prepare_dataset \
    --input data/raw/multi_merged.jsonl

python -m askbuddyx.train.build_training_text

TRAIN_ITERS=2000 \
DATA_LIMIT=5000 \
OUTPUT_DIR="outputs/adapters/v2.0-multi" \
make train
```

### 5.3 Dataset Quality Control

```python
# scripts/validate_dataset.py
import json
from pathlib import Path

def validate_dataset(file_path):
    """Validate dataset quality"""
    
    issues = []
    stats = {
        "total": 0,
        "empty_instruction": 0,
        "empty_output": 0,
        "too_short": 0,
        "too_long": 0,
        "invalid_json": 0
    }
    
    with open(file_path) as f:
        for i, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                stats["total"] += 1
                
                # Check required fields
                if not item.get("instruction", "").strip():
                    stats["empty_instruction"] += 1
                    issues.append(f"Line {i}: Empty instruction")
                
                if not item.get("output", "").strip():
                    stats["empty_output"] += 1
                    issues.append(f"Line {i}: Empty output")
                
                # Check length
                output_len = len(item.get("output", ""))
                if output_len < 10:
                    stats["too_short"] += 1
                elif output_len > 4000:
                    stats["too_long"] += 1
                    issues.append(f"Line {i}: Output too long ({output_len} chars)")
                
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                issues.append(f"Line {i}: Invalid JSON")
    
    # Print report
    print(f"\n=== Dataset Validation Report ===")
    print(f"File: {file_path}")
    print(f"Total examples: {stats['total']}")
    print(f"Empty instructions: {stats['empty_instruction']}")
    print(f"Empty outputs: {stats['empty_output']}")
    print(f"Too short: {stats['too_short']}")
    print(f"Too long: {stats['too_long']}")
    print(f"Invalid JSON: {stats['invalid_json']}")
    
    if issues:
        print(f"\n=== Issues Found ({len(issues)}) ===")
        for issue in issues[:10]:  # Show first 10
            print(issue)
    
    return stats, issues

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/multi_merged.jsonl"
    validate_dataset(file_path)
```

---

## 6. Approach 4: Incremental Fine-Tuning {#approach-4-incremental}

**Best for**: Continuous learning from production feedback

### 6.1 Concept

Start with a trained adapter and continue training on new data:

```
Base Model → Adapter v1.0 (50 iters) → Adapter v1.1 (+ 100 iters on new data)
```

### 6.2 Step-by-Step: Incremental Training

#### Step 1: Collect Production Feedback

```python
# scripts/collect_feedback.py
import json
from datetime import datetime
from pathlib import Path

class FeedbackCollector:
    def __init__(self, feedback_file="data/feedback/production.jsonl"):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    def add_feedback(self, prompt, response, rating, correction=None):
        """Add user feedback"""
        
        feedback = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "response": response,
            "rating": rating,  # 1-5 stars
            "correction": correction,  # Optional improved response
        }
        
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback) + '\n')
    
    def export_training_data(self, min_rating=4):
        """Export high-quality feedback as training data"""
        
        training_data = []
        
        with open(self.feedback_file) as f:
            for line in f:
                item = json.loads(line)
                
                # Only use high-rated responses
                if item["rating"] >= min_rating:
                    record = {
                        "instruction": item["prompt"],
                        "input": "",
                        "output": item["correction"] or item["response"]
                    }
                    training_data.append(record)
        
        # Write to training file
        output_file = Path("data/custom/production_feedback.jsonl")
        with open(output_file, 'w') as f:
            for record in training_data:
                f.write(json.dumps(record) + '\n')
        
        print(f"Exported {len(training_data)} examples to {output_file}")
        return output_file

# Usage in webui/app.py
collector = FeedbackCollector()

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    collector.add_feedback(
        prompt=data['prompt'],
        response=data['response'],
        rating=data['rating'],
        correction=data.get('correction')
    )
    return jsonify({"status": "success"})
```

#### Step 2: Prepare Incremental Dataset

```bash
# Export production feedback
python scripts/collect_feedback.py export

# Merge with existing data (optional)
python scripts/merge_datasets.py \
    --base data/raw/codealpaca.jsonl \
    --custom data/custom/production_feedback.jsonl \
    --output data/raw/incremental.jsonl

# Prepare for training
python -m askbuddyx.train.prepare_dataset \
    --input data/raw/incremental.jsonl
```

#### Step 3: Continue Training from Existing Adapter

```python
# Modify train/run_lora.py to support continuing from adapter

def run_lora_training(
    model_id=MODEL_ID,
    data_dir="data/training_ready",
    output_dir="outputs/adapters/dev",
    iters=50,
    continue_from=None,  # NEW: Path to existing adapter
    **kwargs
):
    """Run LoRA training, optionally continuing from existing adapter"""
    
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", model_id,
        "--train",
        "--data", data_dir,
        "--iters", str(iters),
        "--adapter-path", output_dir,
    ]
    
    # If continuing from existing adapter
    if continue_from:
        # Copy existing adapter to output dir first
        import shutil
        if Path(continue_from).exists():
            print(f"Continuing training from {continue_from}")
            shutil.copytree(continue_from, output_dir, dirs_exist_ok=True)
    
    # Add other parameters...
    subprocess.run(cmd, check=True)
```

#### Step 4: Train Incrementally

```bash
# Continue from v1.0 adapter
python -m askbuddyx.train.run_lora \
    --continue-from outputs/adapters/v1.0 \
    --output outputs/adapters/v1.1-incremental \
    --iters 200

# Or use environment variable
CONTINUE_FROM="outputs/adapters/v1.0" \
OUTPUT_DIR="outputs/adapters/v1.1-incremental" \
TRAIN_ITERS=200 \
make train
```

### 6.3 Incremental Training Best Practices

1. **Small Batches**: Add 100-500 examples at a time
2. **Lower Learning Rate**: Use 1e-5 instead of 2e-5 for fine-tuning
3. **Fewer Iterations**: 100-200 iterations per increment
4. **Frequent Evaluation**: Test after each increment
5. **Version Control**: Keep all intermediate adapters

---

## 7. Version Management Strategy {#version-management}

### 7.1 Semantic Versioning for Adapters

```
v<major>.<minor>.<patch>-<tag>

Examples:
- v1.0.0        # Initial release
- v1.1.0        # More iterations
- v1.2.0        # Custom data added
- v2.0.0        # New dataset
- v2.1.0-beta   # Testing new approach
```

### 7.2 Directory Structure

```
outputs/
├── adapters/
│   ├── dev/              # Development (50 iters)
│   ├── v1.0.0/           # Production release
│   ├── v1.1.0/           # 1000 iterations
│   ├── v1.2.0-custom/    # + custom data
│   ├── v2.0.0-multi/     # + multiple datasets
│   └── v2.1.0-beta/      # Testing
└── experiments/
    ├── exp-001-lr/       # Learning rate experiments
    ├── exp-002-rank/     # Rank experiments
    └── exp-003-data/     # Data experiments
```

### 7.3 Metadata Tracking

```python
# Enhanced run_meta.json
{
    "version": "v1.2.0-custom",
    "base_model": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "datasets": [
        {"name": "code-alpaca-20k", "examples": 2000},
        {"name": "custom", "examples": 150}
    ],
    "hyperparameters": {
        "iters": 1000,
        "rank": 8,
        "alpha": 16,
        "learning_rate": 2e-5,
        "dropout": 0.05
    },
    "metrics": {
        "final_train_loss": 0.245,
        "final_val_loss": 0.298,
        "sanity_prompts": "10/10",
        "codegen_tests": "2/2"
    },
    "training_time": "45m 23s",
    "timestamp": "2025-12-29T18:00:00Z",
    "git_commit": "abc123def",
    "notes": "Added custom examples for async/await"
}
```

---

## 8. Publishing Updates to Hugging Face {#publishing-updates}

### 8.1 Quick Update Workflow

```bash
# Step 1: Train new version
TRAIN_ITERS=1000 OUTPUT_DIR="outputs/adapters/v1.1.0" make train

# Step 2: Evaluate
python -m askbuddyx.eval.run_sanity_prompts --adapter outputs/adapters/v1.1.0
python -m askbuddyx.eval.run_codegen_smoke --adapter outputs/adapters/v1.1.0

# Step 3: Update MODEL_CARD.md
# (See section 8.2)

# Step 4: Create bundle
python -m askbuddyx.publish.make_bundle \
    --adapter outputs/adapters/v1.1.0 \
    --version v1.1.0

# Step 5: Publish to HF
python -m askbuddyx.publish.publish \
    --bundle outputs/hf_bundle \
    --version v1.1.0
```

### 8.2 Update MODEL_CARD.md

```markdown
# AskBuddyX

## Version History

### v1.1.0 (2025-12-29)
- **Training**: 1000 iterations (up from 50)
- **Improvements**:
  - Lower training loss: 0.245 (was 0.298)
  - Lower validation loss: 0.298 (was 0.357)
  - Better code generation quality
- **Datasets**: code-alpaca-20k (2000 examples)
- **Evaluation**: 10/10 sanity prompts, 2/2 codegen tests

### v1.0.0 (2025-12-28)
- Initial release
- 50 iterations baseline
- code-alpaca-20k dataset
```

### 8.3 Enhanced Publishing Script

```python
# askbuddyx/publish/publish_versioned.py
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import json

def publish_versioned(
    bundle_dir="outputs/hf_bundle",
    repo_id="salakash/AskBuddyX",
    version="v1.1.0",
    create_branch=True
):
    """Publish adapter with version tagging"""
    
    api = HfApi()
    
    # Create repo if doesn't exist
    create_repo(repo_id, exist_ok=True, repo_type="model")
    
    # Upload to main branch
    print(f"Uploading {version} to {repo_id}...")
    api.upload_folder(
        folder_path=bundle_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Release {version}"
    )
    
    # Create version branch
    if create_branch:
        try:
            api.create_branch(
                repo_id=repo_id,
                branch=version,
                repo_type="model"
            )
            print(f"Created branch: {version}")
        except Exception as e:
            print(f"Branch may already exist: {e}")
    
    # Create release tag
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=version,
            repo_type="model",
            tag_message=f"Release {version}"
        )
        print(f"Created tag: {version}")
    except Exception as e:
        print(f"Tag may already exist: {e}")
    
    print(f"\n✅ Published {version} to https://huggingface.co/{repo_id}")
    print(f"   Main: https://huggingface.co/{repo_id}")
    print(f"   Version: https://huggingface.co/{repo_id}/tree/{version}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="outputs/hf_bundle")
    parser.add_argument("--version", required=True)
    parser.add_argument("--repo", default="salakash/AskBuddyX")
    args = parser.parse_args()
    
    publish_versioned(args.bundle, args.repo, args.version)
```

### 8.4 Makefile Updates

```makefile
# Add to Makefile

# Publish with version
publish-version:
	@echo "Publishing version $(VERSION)..."
	python -m askbuddyx.publish.make_bundle \
		--adapter outputs/adapters/$(VERSION) \
		--version $(VERSION)
	python -m askbuddyx.publish.publish_versioned \
		--bundle outputs/hf_bundle \
		--version $(VERSION)

# Example usage:
# make publish-version VERSION=v1.1.0
```

---

## 9. Evaluation & Quality Control {#evaluation}

### 9.1 Comprehensive Evaluation Suite

```python
# askbuddyx/eval/comprehensive_eval.py
import json
from pathlib import Path
from mlx_lm import load, generate
import time

class ComprehensiveEvaluator:
    def __init__(self, model_id, adapter_path):
        print(f"Loading model: {model_id}")
        print(f"Loading adapter: {adapter_path}")
        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)
    
    def run_all_tests(self):
        """Run complete evaluation suite"""
        
        results = {
            "sanity_prompts": self.test_sanity_prompts(),
            "codegen": self.test_codegen(),
            "response_format": self.test_response_format(),
            "edge_cases": self.test_edge_cases(),
            "performance": self.test_performance(),
        }
        
        return results
    
    def test_sanity_prompts(self):
        """Test basic functionality"""
        prompts = [
            "Write a hello world function",
            "Create a fibonacci function",
            "Implement binary search",
        ]
        
        passed = 0
        for prompt in prompts:
            response = generate(self.model, self.tokenizer, prompt, max_tokens=200)
            if response and len(response) > 10:
                passed += 1
        
        return {"passed": passed, "total": len(prompts)}
    
    def test_response_format(self):
        """Test if responses follow Solution/Usage/Sanity test format"""
        prompt = "Write a function to reverse a string"
        response = generate(self.model, self.tokenizer, prompt, max_tokens=500)
        
        has_solution = "### Solution" in response
        has_usage = "### Usage" in response
        
        return {
            "has_solution": has_solution,
            "has_usage": has_usage,
            "follows_format": has_solution and has_usage
        }
    
    def test_edge_cases(self):
        """Test edge cases"""
        edge_cases = [
            "Write a function with no parameters",
            "Handle empty input",
            "Deal with None values",
        ]
        
        results = []
        for case in edge_cases:
            response = generate(self.model, self.tokenizer, case, max_tokens=300)
            results.append({
                "case": case,
                "response_length": len(response),
                "has_code": "def " in response or "class " in response
            })
        
        return results
    
    def test_performance(self):
        """Test inference speed"""
        prompt = "Write a simple function"
        
        times = []
        for _ in range(5):
            start = time.time()
            generate(self.model, self.tokenizer, prompt, max_tokens=200)
            times.append(time.time() - start)
        
        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times)
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(args.model, args.adapter)
    results = evaluator.run_all_tests()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
```

### 9.2 A/B Testing Framework

```python
# scripts/ab_test.py
from mlx_lm import load, generate
import json

def ab_test(
    model_id="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    adapter_a="outputs/adapters/v1.0.0",
    adapter_b="outputs/adapters/v1.1.0",
    test_prompts_file="data/test_prompts.json"
):
    """Compare two adapter versions"""
    
    # Load both models
    print("Loading Model A...")
    model_a, tokenizer_a = load(model_id, adapter_path=adapter_a)
    
    print("Loading Model B...")
    model_b, tokenizer_b = load(model_id, adapter_path=adapter_b)
    
    # Load test prompts
    with open(test_prompts_file) as f:
        prompts = json.load(f)
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}/{len(prompts)}: {prompt[:50]}...")
        
        # Generate from both
        response_a = generate(model_a, tokenizer_a, prompt, max_tokens=500)
        response_b = generate(model_b, tokenizer_b, prompt, max_tokens=500)
        
        results.append({
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "length_a": len(response_a),
            "length_b": len(response_b),
        })
    
    # Save results
    with open("outputs/ab_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ A/B test complete. Results saved to outputs/ab_test_results.json")
    return results

if __name__ == "__main__":
    ab_test()
```

---

## 10. Production Workflow {#production-workflow}

### 10.1 Complete Production Pipeline

```bash
#!/bin/bash
# scripts/production_pipeline.sh

set -e  # Exit on error

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: ./production_pipeline.sh <version>"
    echo "Example: ./production_pipeline.sh v1.1.0"
    exit 1
fi

echo "=========================================="
echo "AskBuddyX Production Pipeline"
echo "Version: $VERSION"
echo "=========================================="

# Step 1: Prepare data
echo "\n[1/7] Preparing data..."
python -m askbuddyx.train.fetch_codealpaca
python -m askbuddyx.train.prepare_dataset
python -m askbuddyx.train.build_training_text

# Step 2: Train
echo "\n[2/7] Training model..."
TRAIN_ITERS=1000 OUTPUT_DIR="outputs/adapters/$VERSION" make train

# Step 3: Comprehensive evaluation
echo "\n[3/7] Running comprehensive evaluation..."
python -m askbuddyx.eval.comprehensive_eval --adapter "outputs/adapters/$VERSION"

# Step 4: A/B test against previous version
echo "\n[4/7] Running A/B test..."
PREV_VERSION=$(ls -1 outputs/adapters/ | grep -v $VERSION | tail -1)
if [ ! -z "$PREV_VERSION" ]; then
    python scripts/ab_test.py \
        --adapter-a "outputs/adapters/$PREV_VERSION" \
        --adapter-b "outputs/adapters/$VERSION"
fi

# Step 5: Update documentation
echo "\n[5/7] Updating documentation..."
# Update MODEL_CARD.md with new version info
python scripts/update_model_card.py --version $VERSION

# Step 6: Create bundle
echo "\n[6/7] Creating Hugging Face bundle..."
python -m askbuddyx.publish.make_bundle \
    --adapter "outputs/adapters/$VERSION" \
    --version $VERSION

# Step 7: Publish to Hugging Face
echo "\n[7/7] Publishing to Hugging Face..."
read -p "Publish to Hugging Face? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m askbuddyx.publish.publish_versioned \
        --bundle outputs/hf_bundle \
        --version $VERSION
    echo "✅ Published to https://huggingface.co/salakash/AskBuddyX"
else
    echo "⏭️  Skipped publishing"
fi

echo "\n=========================================="
echo "Pipeline complete!"
echo "Version: $VERSION"
echo "Adapter: outputs/adapters/$VERSION"
echo "=========================================="
```

### 10.2 Makefile Integration

```makefile
# Add to Makefile

.PHONY: production-pipeline
production-pipeline:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION not specified"; \
		echo "Usage: make production-pipeline VERSION=v1.1.0"; \
		exit 1; \
	fi
	bash scripts/production_pipeline.sh $(VERSION)

# Example usage:
# make production-pipeline VERSION=v1.1.0
```

---

## 11. Troubleshooting {#troubleshooting}

### 11.1 Common Issues

#### Issue: Training loss not decreasing

**Symptoms:**
- Loss stays flat or increases
- Model outputs gibberish

**Solutions:**
```bash
# 1. Check learning rate (try lower)
LEARNING_RATE=1e-5 make train

# 2. Increase iterations
TRAIN_ITERS=200 make train

# 3. Check data quality
python scripts/validate_dataset.py data/training_ready/train.jsonl

# 4. Try different rank
LORA_RANK=16 make train
```

#### Issue: Overfitting (val_loss > train_loss)

**Symptoms:**
- Validation loss higher than training loss
- Model memorizes training data

**Solutions:**
```bash
# 1. Reduce iterations
TRAIN_ITERS=100 make train

# 2. Increase dropout
LORA_DROPOUT=0.1 make train

# 3. Add more training data
python scripts/merge_datasets.py

# 4. Use lower rank
LORA_RANK=4 make train
```

#### Issue: Out of memory

**Symptoms:**
- Training crashes
- "Out of memory" error

**Solutions:**
```bash
# 1. Reduce batch size (if supported)
BATCH_SIZE=1 make train

# 2. Use smaller dataset
DATA_LIMIT=1000 make train

# 3. Close other applications
# 4. Restart Mac to free memory
```

#### Issue: Slow training

**Symptoms:**
- Training takes too long
- Each iteration > 10 seconds

**Solutions:**
```bash
# 1. Reduce data size
DATA_LIMIT=1000 make train

# 2. Truncate long examples
# Edit build_training_text.py, reduce max_length

# 3. Use fewer iterations for testing
TRAIN_ITERS=50 make train

# 4. Check Activity Monitor for CPU/memory usage
```

#### Issue: Model outputs don't follow format

**Symptoms:**
- No "### Solution" headings
- Responses not structured

**Solutions:**
```bash
# 1. Check system prompt in prompting.py
# 2. Add more examples with correct format to training data
# 3. Increase training iterations
TRAIN_ITERS=500 make train

# 4. Validate training data format
python scripts/validate_dataset.py
```

### 11.2 Debugging Tools

```python
# scripts/debug_training.py
import json
from pathlib import Path

def debug_training_data(file_path="data/training_ready/train.jsonl"):
    """Debug training data issues"""
    
    print(f"Analyzing {file_path}...")
    
    with open(file_path) as f:
        examples = [json.loads(line) for line in f]
    
    print(f"\nTotal examples: {len(examples)}")
    
    # Check text lengths
    lengths = [len(ex["text"]) for ex in examples]
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Avg length: {sum(lengths) / len(lengths):.0f}")
    
    # Check for format
    has_solution = sum(1 for ex in examples if "### Solution" in ex["text"])
    has_usage = sum(1 for ex in examples if "### Usage" in ex["text"])
    
    print(f"\nFormat compliance:")
    print(f"Has '### Solution': {has_solution}/{len(examples)} ({has_solution/len(examples)*100:.1f}%)")
    print(f"Has '### Usage': {has_usage}/{len(examples)} ({has_usage/len(examples)*100:.1f}%)")
    
    # Show sample
    print(f"\n=== Sample Example ===")
    print(examples[0]["text"][:500])
    print("...")

if __name__ == "__main__":
    debug_training_data()
```

---

## Summary: Recommended Workflow

### For Quick Improvements (1-2 hours)
```bash
# Increase iterations
TRAIN_ITERS=1000 OUTPUT_DIR="outputs/adapters/v1.1.0" make train
python -m askbuddyx.eval.comprehensive_eval --adapter outputs/adapters/v1.1.0
make publish-version VERSION=v1.1.0
```

### For Domain-Specific Knowledge (2-4 hours)
```bash
# Create custom data
vim data/custom/my_examples.jsonl

# Merge and train
python scripts/merge_datasets.py
TRAIN_ITERS=1000 OUTPUT_DIR="outputs/adapters/v1.2.0-custom" make train
python -m askbuddyx.eval.comprehensive_eval --adapter outputs/adapters/v1.2.0-custom
make publish-version VERSION=v1.2.0-custom
```

### For Major Upgrades (4-8 hours)
```bash
# Fetch additional datasets
python -m askbuddyx.train.fetch_additional_dataset --dataset mbpp --limit 500

# Merge all datasets
python scripts/merge_multiple_datasets.py

# Train with more data and iterations
DATA_LIMIT=5000 TRAIN_ITERS=2000 OUTPUT_DIR="outputs/adapters/v2.0.0-multi" make train

# Comprehensive evaluation
python -m askbuddyx.eval.comprehensive_eval --adapter outputs/adapters/v2.0.0-multi
python scripts/ab_test.py --adapter-a outputs/adapters/v1.2.0-custom --adapter-b outputs/adapters/v2.0.0-multi

# Publish
make publish-version VERSION=v2.0.0-multi
```

### For Production Pipeline (Automated)
```bash
# One command for everything
make production-pipeline VERSION=v1.1.0
```

---

**End of Guide**

For questions or issues, refer to:
- Main README: `README.md`
- Architecture Doc: `docs/AskBuddyX_Architecture_and_Engineering.md`
- Model Card: `MODEL_CARD.md`