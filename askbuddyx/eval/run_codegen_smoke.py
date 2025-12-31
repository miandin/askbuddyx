#!/usr/bin/env python3
"""Run code generation smoke tests."""

import argparse
import sys

from mlx_lm import generate, load

from askbuddyx.config import ADAPTER_DIR, MODEL_ID
from askbuddyx.prompting import SYSTEM_PROMPT, format_chat

CODE_PROMPTS = [
    "Write a Python function called 'add' that takes two numbers and returns their sum",
    "Write a Python function called 'is_even' that checks if a number is even",
]


def extract_code(response: str) -> str:
    """Extract code from response (simple heuristic)."""
    lines = response.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        # Look for code blocks or function definitions
        if line.strip().startswith("```python"):
            in_code = True
            continue
        elif line.strip().startswith("```"):
            in_code = False
            continue
        elif line.strip().startswith("def ") or in_code:
            code_lines.append(line)
            in_code = True
        elif in_code and line.strip():
            code_lines.append(line)

    return "\n".join(code_lines)


def run_codegen_smoke(
    model_id: str = MODEL_ID,
    adapter_path: str = None,
    max_tokens: int = 512,
):
    """
    Run code generation smoke tests.

    Args:
        model_id: Model ID to load
        adapter_path: Path to adapter (optional)
        max_tokens: Maximum tokens to generate
    """
    print("=" * 60)
    print("Running Code Generation Smoke Tests")
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

    # Run tests
    passed = 0
    failed = 0

    for i, prompt in enumerate(CODE_PROMPTS, 1):
        print(f"[{i}/{len(CODE_PROMPTS)}] Prompt: {prompt}")

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

            print(f"  Generated response ({len(response)} chars)")

            # Extract code
            code = extract_code(response)

            if not code:
                print("  ✗ No code found in response")
                failed += 1
                continue

            print(f"  Extracted code ({len(code)} chars)")

            # Try to compile
            try:
                compile(code, "<generated>", "exec")
                print("  ✓ Code compiles successfully")
                passed += 1
            except SyntaxError as e:
                print(f"  ✗ Syntax error: {e}")
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(CODE_PROMPTS)}")
    print(f"Failed: {failed}/{len(CODE_PROMPTS)}")

    if failed > 0:
        print("\n⚠️  Some tests failed!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Run code generation smoke tests")
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
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    args = parser.parse_args()
    run_codegen_smoke(
        model_id=args.model,
        adapter_path=args.adapter,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

