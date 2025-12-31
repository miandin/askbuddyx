#!/bin/bash
# Production pipeline for AskBuddyX

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
echo ""
echo "[1/7] Preparing data..."
python -m askbuddyx.train.fetch_codealpaca
python -m askbuddyx.train.prepare_dataset
python -m askbuddyx.train.build_training_text

# Step 2: Train
echo ""
echo "[2/7] Training model..."
TRAIN_ITERS=1000 OUTPUT_DIR="outputs/adapters/$VERSION" make train

# Step 3: Comprehensive evaluation
echo ""
echo "[3/7] Running evaluation..."
python -m askbuddyx.eval.run_sanity_prompts --adapter "outputs/adapters/$VERSION"
python -m askbuddyx.eval.run_codegen_smoke --adapter "outputs/adapters/$VERSION"

# Step 4: A/B test against previous version
echo ""
echo "[4/7] Checking for previous versions..."
PREV_VERSION=$(ls -1 outputs/adapters/ 2>/dev/null | grep -v $VERSION | grep -v dev | tail -1 || echo "")
if [ ! -z "$PREV_VERSION" ]; then
    echo "Found previous version: $PREV_VERSION"
    echo "A/B testing available - run manually: python scripts/ab_test.py"
else
    echo "No previous version found, skipping A/B test"
fi

# Step 5: Update documentation
echo ""
echo "[5/7] Documentation ready for manual update..."
echo "Please update MODEL_CARD.md with version $VERSION details"

# Step 6: Create bundle
echo ""
echo "[6/7] Creating Hugging Face bundle..."
python -m askbuddyx.publish.make_bundle

# Step 7: Publish to Hugging Face
echo ""
echo "[7/7] Ready to publish..."
read -p "Publish to Hugging Face? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m askbuddyx.publish.publish
    echo "✅ Published to https://huggingface.co/salakash/AskBuddyX"
else
    echo "⏭️  Skipped publishing"
    echo "To publish later, run: make publish"
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Version: $VERSION"
echo "Adapter: outputs/adapters/$VERSION"
echo "=========================================="

# Made with Bob
