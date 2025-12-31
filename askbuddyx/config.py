"""Configuration for AskBuddyX."""

import os

# Model configuration
MODEL_ID = os.getenv("MODEL_ID", "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
HF_BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Dataset configuration
DATASET_ID = "flwrlabs/code-alpaca-20k"
DATA_LIMIT = int(os.getenv("DATA_LIMIT", "2000"))

# Training configuration
TRAIN_ITERS = int(os.getenv("TRAIN_ITERS", "50"))
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5

# Serving configuration
SERVED_MODEL_NAME = "AskBuddyX"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))

# Directory configuration
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapters", "dev")
BUNDLE_DIR = os.path.join(OUTPUT_DIR, "hf_bundle")

# Hugging Face configuration
HF_REPO = "salakash/AskBuddyX"

# Data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TRAINING_READY_DIR = os.path.join(DATA_DIR, "training_ready")

