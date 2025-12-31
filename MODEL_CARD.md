---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
- code
- coding-assistant
- mlx
- lora
- qwen2.5
language:
- en
pipeline_tag: text-generation
---

# AskBuddyX

AskBuddyX is a practical coding assistant fine-tuned with LoRA on the code-alpaca-20k dataset. It provides runnable-first responses with structured sections for Solution, Usage, and Sanity Tests.

## Model Details

- **Base Model**: [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
- **MLX Weights**: [mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit)
- **Training Dataset**: [flwrlabs/code-alpaca-20k](https://huggingface.co/datasets/flwrlabs/code-alpaca-20k)
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Framework**: MLX (Apple Silicon optimized)
- **License**: Apache-2.0

## Intended Use

AskBuddyX is designed for:
- Code generation and completion
- Programming assistance and tutoring
- Quick prototyping and examples
- Learning programming concepts

### Response Format

When asked for code, AskBuddyX structures responses with:

1. **Solution**: The main implementation
2. **Usage**: A minimal runnable example
3. **Sanity test**: A tiny test snippet (when appropriate)

This format ensures responses are immediately actionable and testable.

## Training Details

- **Dataset Size**: 2,000 examples (configurable)
- **Training Iterations**: 50 (configurable)
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Learning Rate**: 2e-5
- **Hardware**: Apple Silicon M1 with 32GB RAM

### Data Processing

The training data underwent:
1. Secret redaction (API keys, private keys, tokens)
2. Deduplication by content hash
3. Train/validation split (98/2)
4. Deterministic truncation for efficiency

## Usage

### Installation

```bash
pip install mlx-lm
```

### Running the Server

```bash
python -m mlx_lm.server \
  --model mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit \
  --adapter-path salakash/AskBuddyX \
  --host 127.0.0.1 \
  --port 8080
```

### API Example

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "AskBuddyX",
    "messages": [
      {"role": "user", "content": "Write a Python function to add two numbers"}
    ],
    "max_tokens": 256
  }'
```

### Python Example

```python
from mlx_lm import load, generate

# Load model with adapter
model, tokenizer = load(
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    adapter_path="salakash/AskBuddyX"
)

# Generate response
prompt = "Write a Python function to reverse a string"
response = generate(model, tokenizer, prompt=prompt, max_tokens=256)
print(response)
```

## Limitations

- **Model Size**: 0.5B parameters - suitable for quick tasks but not complex reasoning
- **Context Length**: Limited by base model's context window
- **Domain**: Primarily trained on Python code examples
- **Hardware**: Optimized for Apple Silicon; may not perform optimally on other platforms
- **Accuracy**: May generate incorrect or insecure code; always review outputs

## Ethical Considerations

- **Code Review**: Always review generated code before use in production
- **Security**: Do not use for security-critical applications without thorough review
- **Bias**: May reflect biases present in training data
- **Attribution**: Generated code should be reviewed for licensing implications

## Attribution

This model is built upon:

1. **Base Model**: Qwen/Qwen2.5-Coder-0.5B-Instruct
   - License: Apache-2.0
   - Authors: Qwen Team, Alibaba Cloud
   - No endorsement by original authors is implied

2. **MLX Conversion**: mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit
   - Converted for Apple Silicon optimization
   - Community contribution

3. **Training Dataset**: flwrlabs/code-alpaca-20k
   - License: Apache-2.0
   - Based on Stanford Alpaca methodology
   - No endorsement by dataset authors is implied

## Citation

If you use AskBuddyX in your research or applications, please cite:

```bibtex
@misc{askbuddyx2024,
  title={AskBuddyX: A Practical Coding Assistant},
  author={Kashif Salahuddin},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/salakash/AskBuddyX}}
}
```

## Contact

- Repository: [github.com/salakash/AskBuddyX](https://github.com/salakash/AskBuddyX)
- Issues: [github.com/salakash/AskBuddyX/issues](https://github.com/salakash/AskBuddyX/issues)

## Disclaimer

This adapter is provided "as is" without warranty. The authors are not responsible for any damages or issues arising from its use. Always review and test generated code before deployment.