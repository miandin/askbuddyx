---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
- code
- coding-assistant
- lora
- mlx
- apple-silicon
- qwen2.5
datasets:
- flwrlabs/code-alpaca-20k
- m-a-p/Code-Feedback
library_name: mlx-lm
pipeline_tag: text-generation
---
**Developed By Kashif Salahuddin & Samiya Kashif**
## 1. Executive Summary

**AskBuddyX** is a specialized coding assistant built as a LoRA (Low-Rank Adaptation) adapter for the Qwen2.5-Coder-0.5B-Instruct base model. Unlike generic coding assistants, AskBuddyX implements a "runnable-first" philosophy: when users request code, responses are structured with clear **Solution**, **Usage**, and **Sanity test** sections, ensuring developers receive immediately executable code with minimal friction.

### What AskBuddyX Is

- **A LoRA adapter** Trained on code-alpaca-20k dataset
- **OpenAI-compatible API** for local inference
- **Lightweight distribution** (~12MB adapter vs. multi-GB full models)
- **Production-engineered** with automated pipelines, evaluation, and publishing

## Why AskBuddyX

AskBuddyX is built for a simple, practical goal: **deliver the same outcome with fewer lines of code**.

Most coding assistants tend to “over-achieve” by producing large, multi-step solutions—even when a smaller, clearer implementation would do. That extra code isn’t free: it increases review effort, maintenance cost, and the surface area where defects can hide. 

**Too Much Code, Too Fast** Teams everywhere are seeing a huge jump in the number of lines of code (LOC). Developers—from interns to seniors—are suddenly writing **5 to 7 times more** than before. At first, it looks like higher productivity. In reality, it often means more bugs.

There’s a long-standing rule in software engineering:

> “The more lines of code you have, the higher your probability of introducing bugs.”

The industry’s oldest truth still stands: the more code you have, the more things can go wrong. And AI-generated code tends to be **verbose and repetitive**, which can inflate LOC without adding real value.

AskBuddyX is designed for teams that value **minimalism, clarity, and correctness** over volume.


### What makes AskBuddyX different

* **Minimal LoC by default**
  AskBuddyX is optimized to **minimize lines of code while preserving behavior**—it prefers the smallest correct solution that meets the user’s objective. 

* **Internal governance behavior**
  The model follows a lightweight internal “governance layer” in its response style: avoid unnecessary scaffolding, avoid over-abstraction, keep code focused, and don’t introduce additional complexity that doesn’t improve the result. The governance layer sits between the user request and the model’s final output to enforce **minimalism as a constraint**. It evaluates candidate solutions by measuring **lines of code** and selects the smallest implementation that still satisfies the original requirements. If a shorter variant fails, it automatically falls back to the next-smallest passing candidate, ensuring fewer lines **without sacrificing correctness**.

* **Practical, runnable output**
  When you ask for code, AskBuddyX is tuned toward “runnable-first” answers—clear implementation, a minimal usage example, and a quick sanity check when appropriate.

### Early validation

AskBuddyX was evaluated in a small developer study comparing it with popular coding models on a shared set of tasks. In this pilot, AskBuddyX showed a **clear reduction in lines of code (up to ~30%)** while producing solutions that **executed correctly and achieved the same intended outcomes** under the evaluation harness.

> Note: Results depend on task selection, constraints, and how “equivalence” is measured. We recommend validating on your own codebase and standards.



### Why It Exists

Developers need coding assistance that:
1. Provides **runnable code immediately** without extensive explanation
2. Runs **locally** without cloud dependencies
3. Maintains **small footprint** for fast iteration
4. Offers **structured, predictable responses** for automation

### Who It's For

- **Individual developers** working on their individual projects.
- **Small teams** needing local, private coding assistance
- **Educators** teaching programming with consistent code examples
- **Researchers** experimenting with LoRA fine-tuning on MLX



## Quick Start

### Option 1: Use with MLX 

Install MLX and load the model with adapter:

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

# Load base model with AskBuddyX adapter
model, tokenizer = load(
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    adapter_path="salakash/AskBuddyX"
)

# Generate code
prompt = "Write a Python function to calculate factorial"
response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
print(response)
```

### Option 2: Use with Transformers

```bash
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    trust_remote_code=True
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "salakash/AskBuddyX")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Generate
messages = [{"role": "user", "content": "Write a Python function to add two numbers"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 3: Web UI with MLX

Start an OpenAI-compatible server:

```bash
# Install mlx-lm if not already installed
pip install mlx-lm

# Start server with adapter
mlx_lm.server \
  --model mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit \
  --adapter-path salakash/AskBuddyX \
  --port 8080
```

Then use with any OpenAI-compatible client:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Write a Python function to reverse a string"}
    ],
    "max_tokens": 512
  }'
```

Or use with any OpenAI-compatible web UI like:
- [Open WebUI](https://github.com/open-webui/open-webui)
- [LibreChat](https://github.com/danny-avila/LibreChat)
- [ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)

Configure the UI to point to `http://localhost:8080` as the API endpoint.

### Option 4: Hugging Face Inference API

Use directly via Hugging Face's Inference API (requires HF token):

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/salakash/AskBuddyX"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "Write a Python function to check if a number is prime",
    "parameters": {"max_new_tokens": 256}
})
print(output)
```

## Response Format

AskBuddyX provides structured, runnable-first responses:

- **Solution**: The main implementation code
- **Usage**: A minimal runnable example
- **Sanity test**: A tiny test snippet (when appropriate)

## Comparison
AskBuddyX achieved the same objective in **~8-10 lines of code**, while a standard LLM typically produced **22–26 lines** for the equivalent solution.

### AskBuddyX

![alt text](image-1.png)

### Standard Coding Agent

![alt text](image.png)

## Base Model & Dataset

- **Base Model**: [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
- **MLX Weights**: [mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit)
- **Dataset**: [flwrlabs/code-alpaca-20k](https://huggingface.co/datasets/flwrlabs/code-alpaca-20k)
- **Dataset**: [m-a-p/Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)

## License

This project publishes only adapter artifacts and configuration. The base model and dataset have their own licenses:

- Base Model: Apache-2.0 (Qwen/Qwen2.5-Coder-0.5B-Instruct)
- Dataset: Apache-2.0 (flwrlabs/code-alpaca-20k)

See `LICENSE-THIRD-PARTY.md` for complete attribution.

## Acknowledgments

- Qwen team for the excellent base model
- MLX community for the Apple Silicon optimizations
- flwrlabs for the code-alpaca-20k dataset
- Multimodel Art Projection for m-a-p/Code-Feedback