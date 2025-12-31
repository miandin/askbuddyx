#!/usr/bin/env python3
"""Run sanity check prompts to verify model functionality."""

import argparse
import sys

from mlx_lm import generate, load

from askbuddyx.config import ADAPTER_DIR, MODEL_ID
from askbuddyx.prompting import SYSTEM_PROMPT, format_chat

SANITY_PROMPTS = [
    "Write a Python function to add two numbers",
    "How do I reverse a string in Python?",
    "Create a simple hello world function",
    "Write a function to check if a number is even",
    "How to read a file in Python?",
    "Write a function to find the maximum in a list",
    "Create a function to calculate factorial",
    "How to sort a list in Python?",
    "Write a function to check if a string is palindrome",
    "Create a simple class in Python",
]


def run_sanity_prompts(
    model_id: str = MODEL_ID,
    adapter_path: str = None,
    max_tokens: int = 256,
):
    """
    Run sanity check prompts.

    Args:
        model_id: Model ID to load
        adapter_path: Path to adapter (optional)
        max_tokens: Maximum tokens to generate
    """
    print("=" * 60)
    print("Running Sanity Prompts")
    print("=" * 60)
    print(f"Model: {model_id}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print()

    # Load model
    print("Loading model...")
    try:
        model, tokenizer = load(model_id, adapter_path=adapter_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print()

    # Run prompts
    passed = 0
    failed = 0

    for i, prompt in enumerate(SANITY_PROMPTS, 1):
        print(f"[{i}/{len(SANITY_PROMPTS)}] Prompt: {prompt}")

        # Format prompt
        formatted_prompt = format_chat(SYSTEM_PROMPT, prompt)

        try:
            # Generate
            response = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            # Check if response is non-empty
            if response and len(response.strip()) > 0:
                print(f"  ✓ Generated {len(response)} characters")
                passed += 1
            else:
                print("  ✗ Empty response")
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(SANITY_PROMPTS)}")
    print(f"Failed: {failed}/{len(SANITY_PROMPTS)}")

    if failed > 0:
        print("\n⚠️  Some prompts failed!")
        sys.exit(1)
    else:
        print("\n✓ All prompts passed!")


def main():
    parser = argparse.ArgumentParser(description="Run sanity check prompts")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=ADAPTER_DIR,
        help=f"Adapter path (default: {ADAPTER_DIR})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )

    args = parser.parse_args()
    run_sanity_prompts(
        model_id=args.model,
        adapter_path=args.adapter,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

