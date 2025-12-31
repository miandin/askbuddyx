#!/usr/bin/env python3
"""Publish adapter bundle to Hugging Face."""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo

from askbuddyx.config import BUNDLE_DIR, HF_REPO


def publish_to_hf(
    bundle_dir: str = BUNDLE_DIR,
    repo_id: str = HF_REPO,
):
    """
    Publish bundle to Hugging Face.

    Args:
        bundle_dir: Directory containing bundle to publish
        repo_id: Hugging Face repository ID
    """
    print("=" * 60)
    print("Publishing to Hugging Face")
    print("=" * 60)
    print(f"Bundle directory: {bundle_dir}")
    print(f"Repository: {repo_id}")
    print()

    # Check bundle exists
    if not os.path.exists(bundle_dir):
        print(f"Error: Bundle directory not found: {bundle_dir}")
        print("Please run 'make bundle' first")
        return False

    # Initialize API
    print("Initializing Hugging Face API...")
    try:
        api = HfApi()
    except Exception as e:
        print(f"Error initializing API: {e}")
        print()
        print("Please ensure you are authenticated:")
        print("  hf auth login")
        print("or set HF_TOKEN environment variable")
        return False

    print()

    # Create repo if it doesn't exist
    print(f"Creating repository (if needed): {repo_id}")
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print("✓ Repository ready")
    except Exception as e:
        print(f"Error creating repository: {e}")
        print()
        print("Please check:")
        print("  1. You are authenticated (hf auth login)")
        print("  2. You have permission to create/write to this repo")
        print("  3. The repository name is valid")
        return False

    print()

    # Upload folder
    print("Uploading bundle...")
    try:
        api.upload_folder(
            folder_path=bundle_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        print("✓ Upload complete")
    except Exception as e:
        print(f"Error uploading: {e}")
        print()
        print("If authentication failed, try:")
        print("  hf auth login")
        print("or set HF_TOKEN environment variable")
        return False

    print()
    print("=" * 60)
    print("Published successfully!")
    print("=" * 60)
    print(f"View at: https://huggingface.co/{repo_id}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Publish to Hugging Face")
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default=BUNDLE_DIR,
        help=f"Bundle directory (default: {BUNDLE_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO,
        help=f"Hugging Face repository ID (default: {HF_REPO})",
    )

    args = parser.parse_args()
    success = publish_to_hf(
        bundle_dir=args.bundle_dir,
        repo_id=args.repo_id,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

