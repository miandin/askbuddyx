#!/usr/bin/env python3
"""
Test AskBuddyX inference from Hugging Face
Supports single prompts, batch testing, and performance benchmarking
"""
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from mlx_lm import load, generate
from askbuddyx.config import MODEL_ID, HF_REPO
from huggingface_hub import snapshot_download
import tempfile
import shutil

def load_model_from_hf(repo_id=HF_REPO, base_model=MODEL_ID):
    """Load AskBuddyX model directly from Hugging Face (always fresh download, no cache)"""
    print(f"Loading base model: {base_model}")
    print(f"Downloading adapter from Hugging Face: {repo_id}")
    print("Fetching fresh copy (no cache)...")
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="askbuddyx_adapter_")
    
    try:
        # Download from Hugging Face to temp directory (bypassing cache)
        adapter_path = snapshot_download(
            repo_id,
            repo_type="model",
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md"],
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
            force_download=True
        )
        print(f"✅ Downloaded to: {adapter_path}")
        
        # Load model with adapter
        start_time = time.time()
        model, tokenizer = load(base_model, adapter_path=adapter_path)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f}s\n")
        
        return model, tokenizer, temp_dir
        
    except Exception as e:
        # Cleanup temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Failed to download adapter from Hugging Face: {e}\n"
            f"Please ensure:\n"
            f"1. The model is published: {repo_id}\n"
            f"2. You have internet connection\n"
            f"3. You have HF authentication if repo is private"
        )

def run_inference(model, tokenizer, prompt, max_tokens=500, verbose=True):
    """Run single inference"""
    if verbose:
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    
    inference_time = time.time() - start_time
    
    if verbose:
        print(f"Response:\n{response}\n")
        print(f"{'='*60}")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Tokens: ~{len(response.split())}")
        print(f"{'='*60}\n")
    
    return {
        "prompt": prompt,
        "response": response,
        "inference_time": inference_time,
        "token_count": len(response.split()),
        "timestamp": datetime.now().isoformat()
    }

def run_multiple_inferences(model, tokenizer, prompt, num_runs=5, max_tokens=500):
    """Run inference multiple times for performance testing"""
    print(f"\n{'='*60}")
    print(f"Running {num_runs} inference(s) for prompt:")
    print(f"{prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"{'='*60}\n")
    
    results = []
    times = []
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...")
        result = run_inference(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            verbose=(i == 0)  # Only show details for first run
        )
        results.append(result)
        times.append(result["inference_time"])
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n{'='*60}")
    print(f"Performance Statistics ({num_runs} runs)")
    print(f"{'='*60}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Min time: {min_time:.2f}s")
    print(f"Max time: {max_time:.2f}s")
    print(f"Std deviation: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f}s")
    print(f"{'='*60}\n")
    
    return results

def run_batch_prompts(model, tokenizer, prompts_file, max_tokens=500):
    """Run inference on multiple prompts from a file"""
    print(f"Loading prompts from: {prompts_file}")
    
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(prompts)} prompts\n")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Prompt {i}/{len(prompts)}")
        print(f"{'='*60}")
        
        result = run_inference(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            verbose=True
        )
        results.append(result)
    
    return results

def save_results(results, output_file="outputs/inference_results.json"):
    """Save inference results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")

def interactive_mode(model, tokenizer, max_tokens=500):
    """Interactive prompt mode"""
    print(f"\n{'='*60}")
    print("Interactive Mode")
    print("Type your prompts (or 'quit' to exit)")
    print(f"{'='*60}\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            if not prompt:
                continue
            
            run_inference(
                model, tokenizer, prompt,
                max_tokens=max_tokens,
                verbose=True
            )
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Test AskBuddyX inference from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt
  python scripts/test_hf_inference.py --prompt "Write a hello world function"
  
  # Run same prompt 10 times (performance test)
  python scripts/test_hf_inference.py --prompt "Write fibonacci" --runs 10
  
  # Batch prompts from file
  python scripts/test_hf_inference.py --batch prompts.txt
  
  # Interactive mode
  python scripts/test_hf_inference.py --interactive
  
  # Use custom repo
  python scripts/test_hf_inference.py --repo salakash/AskBuddyX --prompt "Test"
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--repo",
        default=HF_REPO,
        help=f"Hugging Face repo ID (default: {HF_REPO})"
    )
    parser.add_argument(
        "--base-model",
        default=MODEL_ID,
        help=f"Base model ID (default: {MODEL_ID})"
    )
    
    # Inference mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--prompt",
        help="Single prompt to test"
    )
    mode_group.add_argument(
        "--batch",
        help="File containing prompts (one per line)"
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive prompt mode"
    )
    
    # Inference parameters
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run inference (for performance testing)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate (default: 500)"
    )
    parser.add_argument(
        "--output",
        default="outputs/inference_results.json",
        help="Output file for results (default: outputs/inference_results.json)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    args = parser.parse_args()
    
    # Load model (returns temp_dir for cleanup)
    model, tokenizer, temp_dir = load_model_from_hf(args.repo, args.base_model)
    
    try:
        # Run inference based on mode
        results = []
        
        if args.prompt:
            if args.runs > 1:
                results = run_multiple_inferences(
                    model, tokenizer, args.prompt,
                    num_runs=args.runs,
                    max_tokens=args.max_tokens
                )
            else:
                result = run_inference(
                    model, tokenizer, args.prompt,
                    max_tokens=args.max_tokens,
                    verbose=True
                )
                results = [result]
        
        elif args.batch:
            results = run_batch_prompts(
                model, tokenizer, args.batch,
                max_tokens=args.max_tokens
            )
        
        elif args.interactive:
            interactive_mode(
                model, tokenizer,
                max_tokens=args.max_tokens
            )
            return  # Don't save results in interactive mode
        
        # Save results
        if results and not args.no_save:
            save_results(results, args.output)
    
    finally:
        # Cleanup: Remove temporary download directory
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✅ Cleanup complete")

if __name__ == "__main__":
    main()

# Made with Bob
