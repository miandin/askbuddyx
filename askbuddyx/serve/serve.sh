#!/bin/bash
# Start OpenAI-compatible server for AskBuddyX

set -e

# Load configuration from environment or use defaults
MODEL_ID="${MODEL_ID:-mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/adapters/dev}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-AskBuddyX}"

echo "========================================"
echo "Starting AskBuddyX Server"
echo "========================================"
echo "Model: $MODEL_ID"
echo "Adapter: $ADAPTER_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo ""

# Check if adapter exists
if [ -d "$ADAPTER_PATH" ]; then
    echo "✓ Adapter found at $ADAPTER_PATH"
    ADAPTER_ARG="--adapter-path $ADAPTER_PATH"
else
    echo "⚠️  No adapter found at $ADAPTER_PATH, using base model only"
    ADAPTER_ARG=""
fi

echo ""

# Try mlx_lm.server first (built-in with mlx-lm)
if python3 -m mlx_lm.server --help &>/dev/null; then
    echo "Using mlx_lm.server..."
    echo ""
    
    # Start server
    python3 -m mlx_lm.server \
        --model "$MODEL_ID" \
        $ADAPTER_ARG \
        --host "$HOST" \
        --port "$PORT" &
    
    SERVER_PID=$!
    
    # Wait for server to start
    echo "Waiting for server to start..."
    sleep 5
    
    echo ""
    echo "========================================"
    echo "Server Started!"
    echo "========================================"
    echo "Server is running on http://$HOST:$PORT"
    echo ""
    echo "Test with curl:"
    echo ""
    echo "curl http://$HOST:$PORT/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{"
    echo "    \"model\": \"$SERVED_MODEL_NAME\","
    echo "    \"messages\": ["
    echo "      {\"role\": \"user\", \"content\": \"Write a Python function to add two numbers\"}"
    echo "    ],"
    echo "    \"max_tokens\": 256"
    echo "  }'"
    echo ""
    echo "Press Ctrl+C to stop the server"
    
    # Wait for server process
    wait $SERVER_PID
else
    echo "❌ Error: mlx_lm.server not found"
    echo ""
    echo "Please install mlx-lm:"
    echo "  pip install mlx-lm"
    echo ""
    exit 1
fi

