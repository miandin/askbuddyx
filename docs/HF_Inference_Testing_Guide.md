# AskBuddyX Hugging Face Inference Testing Guide

Quick reference for testing your published AskBuddyX model from Hugging Face.

**Note**: This script always downloads fresh from Hugging Face (no caching) and cleans up after each run.

## Quick Start

### Single Prompt Test
```bash
python scripts/test_hf_inference.py \
    --prompt "Write a hello world function"
```

### Performance Test (Run 10 times)
```bash
python scripts/test_hf_inference.py \
    --prompt "Write a fibonacci function" \
    --runs 10
```

### Batch Testing
```bash
python scripts/test_hf_inference.py \
    --batch data/test_prompts.txt
```

### Interactive Mode
```bash
python scripts/test_hf_inference.py --interactive
```

## All Options

```bash
python scripts/test_hf_inference.py [OPTIONS]

Required (choose one):
  --prompt TEXT          Single prompt to test
  --batch FILE          File with prompts (one per line)
  --interactive         Interactive prompt mode

Optional:
  --repo TEXT           HF repo ID (default: salakash/AskBuddyX)
  --base-model TEXT     Base model ID (default: mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit)
  --runs INT            Number of times to run (default: 1)
  --max-tokens INT      Max tokens to generate (default: 500)
  --temperature FLOAT   Sampling temperature (default: 0.7)
  --output FILE         Output JSON file (default: outputs/inference_results.json)
  --no-save            Don't save results to file
```

## Usage Examples

### 1. Basic Single Prompt
```bash
python scripts/test_hf_inference.py \
    --prompt "Write a function to reverse a string"
```

**Output:**
```
Loading base model: mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit
Loading adapter from: salakash/AskBuddyX
✅ Model loaded in 3.45s

============================================================
Prompt: Write a function to reverse a string
============================================================

Response:
### Solution
def reverse_string(s):
    return s[::-1]

### Usage
text = "hello"
reversed_text = reverse_string(text)
print(reversed_text)  # Output: olleh

### Sanity test
assert reverse_string("hello") == "olleh"
assert reverse_string("") == ""

============================================================
Inference time: 2.34s
Tokens: ~45
============================================================
```

### 2. Performance Benchmarking (10 runs)
```bash
python scripts/test_hf_inference.py \
    --prompt "Create a simple calculator class" \
    --runs 10
```

**Output:**
```
============================================================
Running 10 inference(s) for prompt:
Create a simple calculator class
============================================================

Run 1/10...
[Shows full response for first run]

Run 2/10...
Run 3/10...
...
Run 10/10...

============================================================
Performance Statistics (10 runs)
============================================================
Average time: 2.45s
Min time: 2.31s
Max time: 2.67s
Std deviation: 0.12s
============================================================

✅ Results saved to: outputs/inference_results.json
```

### 3. Batch Testing from File
```bash
# Create test prompts file
cat > my_prompts.txt << 'EOF'
Write a hello world function
Create a fibonacci function
Implement binary search
EOF

# Run batch test
python scripts/test_hf_inference.py --batch my_prompts.txt
```

**Output:**
```
Loading prompts from: my_prompts.txt
Found 3 prompts

============================================================
Prompt 1/3
============================================================
Prompt: Write a hello world function
[Shows response]

============================================================
Prompt 2/3
============================================================
Prompt: Create a fibonacci function
[Shows response]

============================================================
Prompt 3/3
============================================================
Prompt: Implement binary search
[Shows response]

✅ Results saved to: outputs/inference_results.json
```

### 4. Interactive Mode
```bash
python scripts/test_hf_inference.py --interactive
```

**Session:**
```
============================================================
Interactive Mode
Type your prompts (or 'quit' to exit)
============================================================

Prompt: Write a function to check if a number is prime
[Shows response]

Prompt: Create a merge sort implementation
[Shows response]

Prompt: quit
Exiting interactive mode...
```

### 5. Custom Parameters
```bash
python scripts/test_hf_inference.py \
    --prompt "Write a complex algorithm" \
    --max-tokens 1000 \
    --temperature 0.5 \
    --output my_results.json
```

### 6. Test Different Version
```bash
# Test a specific version/branch
python scripts/test_hf_inference.py \
    --repo salakash/AskBuddyX \
    --prompt "Test prompt"
```

## Output Format

Results are saved as JSON:

```json
[
  {
    "prompt": "Write a hello world function",
    "response": "### Solution\ndef hello():\n    print('Hello, World!')\n\n### Usage\nhello()",
    "inference_time": 2.34,
    "token_count": 25,
    "timestamp": "2025-12-29T22:00:00.000Z"
  }
]
```

## Performance Metrics

Typical performance on M1 Mac (32GB RAM):

| Metric | Value |
|--------|-------|
| Model load time | 3-5 seconds (first time) |
| Inference time | 2-3 seconds |
| Tokens/second | ~15-20 |
| Memory usage | ~2-3 GB |

## Tips

1. **First Run**: Model download takes time on first run (cached after)
2. **Performance Testing**: Use `--runs 10` to get reliable averages
3. **Batch Testing**: Use `data/test_prompts.txt` for standard tests
4. **Interactive Mode**: Best for quick experimentation
5. **Save Results**: Results auto-saved to `outputs/inference_results.json`

## Troubleshooting

### Issue: Model not found
```
Error: Repository not found
```
**Solution:** Check repo name and ensure it's published:
```bash
# Verify repo exists
python -c "from huggingface_hub import HfApi; print(HfApi().repo_info('salakash/AskBuddyX'))"
```

### Issue: Out of memory
```
Error: Out of memory
```
**Solution:** Reduce max_tokens or close other applications:
```bash
python scripts/test_hf_inference.py \
    --prompt "Test" \
    --max-tokens 200
```

### Issue: Slow inference
```
Inference taking > 10 seconds
```
**Solution:** Check Activity Monitor, restart Mac, or reduce token count

## How It Works

### Fresh Download Every Time
The script is designed to:
1. **Download fresh** from Hugging Face on every run
2. **No caching** - always gets the latest version
3. **Auto cleanup** - removes temporary files after completion

### Download Process
```
1. Create temporary directory
2. Download adapter from HF to temp dir
3. Load model with adapter
4. Run inference
5. Save results
6. Clean up temporary files
```

### Why No Cache?
- Always test the latest published version
- Avoid stale cached versions
- Ensure consistency across runs
- Clean environment for each test

## Integration with Other Tools

### Use with curl (via server)
```bash
# Start server
make serve

# Test with curl
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "AskBuddyX",
    "messages": [{"role": "user", "content": "Write hello world"}]
  }'
```

## Sample Test Prompts

Use `data/test_prompts.txt` for standard testing:
- Write a hello world function in Python
- Create a function to calculate fibonacci numbers
- Implement binary search algorithm
- Write a REST API endpoint using Flask
- Create a function to reverse a string
- Implement a simple calculator class
- Write a function to check if a number is prime
- Create a merge sort implementation
- Write a function to validate email addresses
- Implement a basic linked list class

## Next Steps

After testing:
1. Review results in `outputs/inference_results.json`
2. Compare with previous versions
3. Update MODEL_CARD.md with performance metrics
4. Share results with team
5. Deploy to production if satisfied

## Related Documentation

- Main README: `README.md`
- Fine-tuning Guide: `docs/Iterative_Fine_Tuning_Guide.md`
- Architecture Doc: `docs/AskBuddyX_Architecture_and_Engineering.md`
- AWS Deployment: `docs/AWS_S3_Deployment_Quick_Guide.md`