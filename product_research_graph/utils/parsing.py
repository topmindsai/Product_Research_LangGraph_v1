"""Shared parsing utilities for LangGraph nodes.

This module contains common functions used across search, filter, and validate nodes:
- Text extraction from LLM messages (handles Responses API format)
- JSON extraction from LLM responses (handles markdown, reasoning text)
"""

import re


def extract_text_from_message(message) -> str:
    """
    Extract text from AIMessage, handling both string and content block formats.

    With output_version="responses/v1", AIMessage.content may be a list of
    content blocks instead of a plain string.

    Args:
        message: An AIMessage or similar message object

    Returns:
        Extracted text content as a string
    """
    # Prefer .text property if available (Responses API convenience)
    if hasattr(message, 'text') and message.text:
        return message.text

    content = message.content

    # If content is already a string, return it
    if isinstance(content, str):
        return content

    # If content is a list of content blocks, extract text
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    return str(content) if content else ""


def extract_json_from_response(raw_results: str) -> str | None:
    """
    Extract JSON from LLM response that may contain reasoning text.

    Handles cases where JSON is:
    1. Wrapped in ```json ... ``` markdown blocks anywhere in the text
    2. Raw JSON object embedded in text
    3. Plain JSON response

    Args:
        raw_results: Raw response string from LLM

    Returns:
        Extracted JSON string, or None if no JSON found
    """
    content = raw_results.strip()

    # Method 1: Extract JSON from markdown code block anywhere in text
    json_match = re.search(r'```json\s*([\s\S]*?)```', content)
    if json_match:
        return json_match.group(1).strip()

    # Method 2: Try plain ``` blocks
    plain_match = re.search(r'```\s*([\s\S]*?)```', content)
    if plain_match:
        extracted = plain_match.group(1).strip()
        # Verify it looks like JSON
        if extracted.startswith('{'):
            return extracted

    # Method 3: Find raw JSON object by locating matching braces
    json_start = content.find('{')
    if json_start >= 0:
        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(content[json_start:], start=json_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return content[json_start:i + 1]

    # Method 4: If content is already valid JSON
    if content.startswith('{'):
        return content

    return None
