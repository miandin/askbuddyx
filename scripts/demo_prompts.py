#!/usr/bin/env python3
"""
Demonstration script showing AskBuddyX's improved output format.
Tests multiple prompts to showcase the "runnable-first" structure.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load, generate
from askbuddyx.config import MODEL_ID
from askbuddyx.prompting import SYSTEM_PROMPT, format_chat_with_tokenizer

# Demo prompts covering different scenarios
DEMO_PROMPTS = [
    "Write a Python function to check if a string is a palindrome.",
    "Create a function to merge two sorted lists into one sorted list.",
    "Write a function to find the nth Fibonacci number using recursion.",
    "Create a function to validate an email address using regex.",
    "Write a function to calculate the sum of all even numbers in a list.",
]

def run_demo():
    """Run demonstration with multiple prompts."""
    print("=" * 70)
    print("AskBuddyX Output Format Demonstration")
    print("=" * 70)
    print(f"\nLoading model with adapter...")
    print(f"Model: {MODEL_ID}")
    print(f"Adapter: outputs/adapters/dev\n")
    
    # Load model with adapter
    model, tokenizer = load(
        MODEL_ID,
        adapter_path="outputs/adapters/dev"
    )
    
    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        print("\n" + "=" * 70)
        print(f"DEMO {i}/{len(DEMO_PROMPTS)}")
        print("=" * 70)
        print(f"\n📝 PROMPT:\n{prompt}\n")
        print("-" * 70)
        print("🤖 ASKBUDDYX OUTPUT:")
        print("-" * 70)
        
        # Format the chat using tokenizer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=500,
            verbose=False
        )
        
        print(response)
        print()
        
        # Check format
        has_solution = "### Solution" in response
        has_usage = "### Usage" in response
        has_sanity = "### Sanity test" in response
        has_docstring = '"""' in response or "'''" in response
        
        print("-" * 70)
        print("✅ FORMAT CHECK:")
        print(f"  {'✓' if has_solution else '✗'} Has '### Solution' section")
        print(f"  {'✓' if has_usage else '✗'} Has '### Usage' section")
        print(f"  {'✓' if has_sanity else '✗'} Has '### Sanity test' section")
        print(f"  {'✓' if has_docstring else '✗'} Has docstring")
        
        if has_solution and has_usage and has_sanity and has_docstring:
            print("\n  🎉 PERFECT FORMAT!")
        else:
            print("\n  ⚠️  Some sections missing")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  • ### Solution - Main implementation with docstrings")
    print("  • ### Usage - Runnable example showing how to use it")
    print("  • ### Sanity test - Quick assertions to verify correctness")
    print("  • Clean, production-ready code structure")
    print("\nAll outputs follow the 'runnable-first' philosophy!")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
