.PHONY: all deps fetch-data prep-data train eval serve bundle publish clean help

# Default target
all: deps fetch-data prep-data train eval

# Install dependencies
deps:
	@echo "Installing dependencies..."
	pip install -U pip
	pip install -e .
	@echo "✓ Dependencies installed"

# Fetch dataset
fetch-data:
	@echo "Fetching dataset..."
	python -m askbuddyx.train.fetch_codealpaca
	@echo "✓ Dataset fetched"

# Prepare dataset
prep-data:
	@echo "Preparing dataset..."
	python -m askbuddyx.train.prepare_dataset
	python -m askbuddyx.train.build_training_text
	@echo "✓ Dataset prepared"

# Train LoRA adapter
train:
	@echo "Training LoRA adapter..."
	python -m askbuddyx.train.run_lora
	@echo "✓ Training complete"

# Run evaluation
eval:
	@echo "Running evaluation..."
	python -m askbuddyx.eval.run_sanity_prompts
	python -m askbuddyx.eval.run_codegen_smoke
	@echo "✓ Evaluation complete"

# Start server
serve:
	@echo "Starting server..."
	@chmod +x askbuddyx/serve/serve.sh
	@bash askbuddyx/serve/serve.sh

# Create bundle for publishing
bundle:
	@echo "Creating bundle..."
	python -m askbuddyx.publish.make_bundle
	@echo "✓ Bundle created"

# Publish to Hugging Face
publish: bundle
	@echo "Publishing to Hugging Face..."
	python -m askbuddyx.publish.publish
	@echo "✓ Published"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/
	rm -rf outputs/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned"

# Run linter
lint:
	@echo "Running linter..."
	ruff check askbuddyx/
	@echo "✓ Linting complete"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/
	@echo "✓ Tests complete"

# Show help
help:
	@echo "AskBuddyX Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Run complete pipeline (deps, fetch, prep, train, eval)"
	@echo "  deps        - Install dependencies"
	@echo "  fetch-data  - Fetch dataset from Hugging Face"
	@echo "  prep-data   - Prepare and clean dataset"
	@echo "  train       - Train LoRA adapter"
	@echo "  eval        - Run evaluation tests"
	@echo "  serve       - Start OpenAI-compatible server"
	@echo "  bundle      - Create Hugging Face bundle"
	@echo "  publish     - Publish bundle to Hugging Face"
	@echo "  clean       - Remove generated files"
	@echo "  lint        - Run code linter"
	@echo "  test        - Run tests"
	@echo "  help        - Show this help message"