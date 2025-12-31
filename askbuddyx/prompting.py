"""Prompting utilities for AskBuddyX."""

SYSTEM_PROMPT = """You are AskBuddyX, a practical coding assistant.

Default style:
- Be code-forward and concise.
- Prefer runnable solutions.

When the user asks for code, respond using these headings (in this order):
### Solution
### Usage
### Sanity test

Rules:
- "Solution" contains the main implementation.
- "Usage" shows the smallest runnable example (how to call it).
- "Sanity test" is a tiny test snippet (often using assert). Include it only when it makes sense.
- Keep explanations to a few lines maximum. Avoid long essays.
- If something is ambiguous, make one conservative assumption and state it briefly.
- Do not invent APIs or library functions. If unsure, say so and offer a safe alternative.
- If the user explicitly requests a different format, follow the user's format."""


def format_chat(system: str, user: str, assistant: str = "") -> str:
    """
    Format a chat conversation into a string.

    Prefers tokenizer.apply_chat_template if available, otherwise uses simple format.

    Args:
        system: System prompt
        user: User message
        assistant: Assistant response (optional)

    Returns:
        Formatted chat string
    """
    # Simple fallback format
    parts = [f"SYSTEM: {system}", f"USER: {user}"]
    if assistant:
        parts.append(f"ASSISTANT: {assistant}")
    return "\n".join(parts)


def format_chat_with_tokenizer(tokenizer, system: str, user: str, assistant: str = ""):
    """
    Format chat using tokenizer's apply_chat_template if available.

    Args:
        tokenizer: Tokenizer with apply_chat_template method
        system: System prompt
        user: User message
        assistant: Assistant response (optional)

    Returns:
        Formatted chat string
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if assistant:
        messages.append({"role": "assistant", "content": assistant})

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=not assistant
            )
    except Exception:
        pass

    # Fallback to simple format
    return format_chat(system, user, assistant)


def build_training_prompt(instruction: str, input_text: str, output: str) -> str:
    """
    Build a training prompt from instruction, input, and output.

    Args:
        instruction: The instruction/task
        input_text: Additional input context (may be empty)
        output: Expected output/response

    Returns:
        Formatted training text
    """
    user_message = instruction
    if input_text and input_text.strip():
        user_message = f"{instruction}\n\nInput:\n{input_text}"

    return format_chat(SYSTEM_PROMPT, user_message, output)

