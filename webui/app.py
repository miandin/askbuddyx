#!/usr/bin/env python3
"""Simple Web UI for AskBuddyX chatbot."""

import os
from flask import Flask, render_template, request, jsonify, stream_with_context, Response
from mlx_lm import load, generate
from askbuddyx.prompting import SYSTEM_PROMPT

app = Flask(__name__)

# Global model and tokenizer
model = None
tokenizer = None
chat_history = []

def load_model():
    """Load the model with adapter."""
    global model, tokenizer
    print("Loading AskBuddyX model...")
    model, tokenizer = load(
        "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        adapter_path="outputs/adapters/dev"
    )
    print("Model loaded successfully!")

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    global chat_history
    
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Add user message to history
    chat_history.append({'role': 'user', 'content': user_message})
    
    # Build messages with system prompt
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + chat_history
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False
    )
    
    # Extract assistant response
    if 'assistant' in response:
        assistant_message = response.split('assistant')[-1].strip()
    else:
        assistant_message = response
    
    # Add to history
    chat_history.append({'role': 'assistant', 'content': assistant_message})
    
    return jsonify({
        'response': assistant_message,
        'history_length': len(chat_history)
    })

@app.route('/clear', methods=['POST'])
def clear():
    """Clear chat history."""
    global chat_history
    chat_history = []
    return jsonify({'status': 'cleared'})

@app.route('/history', methods=['GET'])
def history():
    """Get chat history."""
    return jsonify({'history': chat_history})

if __name__ == '__main__':
    load_model()
    port = 5001
    print("\n" + "="*60)
    print("AskBuddyX Web UI Starting...")
    print(f"Open your browser and go to: http://localhost:{port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)

# Made with Bob
