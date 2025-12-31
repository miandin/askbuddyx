#!/usr/bin/env python3
"""Test if the retrained model produces the desired output format."""

import sys
from mlx_lm import load, generate

def test_model_output():
    """Test the model with a simple coding request."""
    
    print("Loading model with adapter...")
    model, tokenizer = load(
        "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        adapter_path="outputs/adapters/dev"
    )
    
    # Test prompt
    prompt = "Write a Python function to calculate the factorial of a number."
    
    messages = [
        {
            "role": "system",
            "content": """You are AskBuddyX, a practical coding assistant.

Default style:
- Be code-forward and concise.
- Prefer runnable solutions.

When the user asks for code, respond using these headings (in this order):
### Solution
### Usage
### Sanity test

Rules:
- "Solution" contains the main implementation.
- "Usage" shows the smallest runnable example (how to call it).
- "Sanity test" is a tiny test snippet (often using assert). Include it only when it makes sense.
- Keep explanations to a few lines maximum. Avoid long essays.
- If something is ambiguous, make one conservative assumption and state it briefly.
- Do not invent APIs or library functions. If unsure, say so and offer a safe alternative.
- If the user explicitly requests a different format, follow the user's format."""
        },
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"\n{'='*60}")
    print("PROMPT:")
    print(f"{'='*60}")
    print(prompt)
    print(f"\n{'='*60}")
    print("MODEL OUTPUT:")
    print(f"{'='*60}\n")
    
    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=500,
        verbose=False
    )
    
    print(response)
    print(f"\n{'='*60}")
    
    # Check for expected sections
    has_solution = "### Solution" in response
    has_usage = "### Usage" in response
    has_sanity = "### Sanity test" in response
    has_docstring = '"""' in response or "'''" in response
    has_args = "Args:" in response or "Parameters:" in response
    has_returns = "Returns:" in response
    
    print("\nFORMAT CHECK:")
    print(f"{'='*60}")
    print(f"✓ Has '### Solution' section: {has_solution}")
    print(f"✓ Has '### Usage' section: {has_usage}")
    print(f"✓ Has '### Sanity test' section: {has_sanity}")
    print(f"✓ Has docstring: {has_docstring}")
    print(f"✓ Has Args/Parameters: {has_args}")
    print(f"✓ Has Returns: {has_returns}")
    print(f"{'='*60}")
    
    if has_solution and has_usage and has_docstring:
        print("\n✅ SUCCESS: Model output includes expected format!")
        return 0
    else:
        print("\n⚠️  WARNING: Model output missing some expected sections")
        return 1

if __name__ == "__main__":
    sys.exit(test_model_output())
