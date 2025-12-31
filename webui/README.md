# AskBuddyX Web UI

A simple, beautiful web interface for chatting with your AskBuddyX coding assistant.

## Features

- 💬 Clean chat interface
- 🎨 Beautiful gradient design
- 💻 Code syntax highlighting
- 📝 Markdown support
- 🔄 Chat history management
- ⚡ Real-time responses

## Quick Start

### 1. Install Dependencies

```bash
cd webui
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python app.py
```

### 3. Open Your Browser

Navigate to: http://localhost:5000

## Usage

1. Type your coding question in the input box
2. Press Enter or click "Send"
3. Wait for AskBuddyX to generate a response
4. Click "Clear Chat" to start a new conversation

## Example Prompts

- "Write a Python function to reverse a string"
- "How do I read a CSV file in Python?"
- "Create a function to check if a number is prime"
- "Explain list comprehensions in Python"
- "Write a function to merge two sorted lists"

## Features

### Code Formatting
The UI automatically formats code blocks with syntax highlighting:
- Inline code: \`code\`
- Code blocks: \`\`\`python ... \`\`\`

### Chat History
- All messages are stored in memory during the session
- Use "Clear Chat" to reset the conversation
- History is lost when the server restarts

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, edit `app.py` and change:
```python
app.run(host='0.0.0.0', port=5000, debug=False)
```
to a different port, e.g., `port=5001`

### Model Not Loading
Ensure you have the trained adapter in `outputs/adapters/dev/`:
```bash
ls ../outputs/adapters/dev/
# Should show: adapters.safetensors, adapter_config.json, run_meta.json
```

### Slow Responses
First response may be slow as the model loads. Subsequent responses will be faster.

## Architecture

- **Backend**: Flask (Python web framework)
- **Frontend**: Vanilla JavaScript + HTML/CSS
- **Model**: MLX-LM with LoRA adapter
- **Chat Format**: Uses Qwen2.5 chat template

## Customization

### Change Port
Edit `app.py`, line 95:
```python
app.run(host='0.0.0.0', port=YOUR_PORT, debug=False)
```

### Adjust Max Tokens
Edit `app.py`, line 59:
```python
max_tokens=1024,  # Change this value
```

### Modify System Prompt
The system prompt is imported from `askbuddyx.prompting.SYSTEM_PROMPT`.
Edit `../askbuddyx/prompting.py` to customize the assistant's behavior.

## Security Notes

⚠️ This is a development server. For production use:
- Use a production WSGI server (gunicorn, uwsgi)
- Add authentication
- Enable HTTPS
- Add rate limiting
- Sanitize user inputs

## License

Same as AskBuddyX project (Apache-2.0)