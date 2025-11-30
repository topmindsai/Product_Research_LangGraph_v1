"""Filter node for the Product Research workflow."""

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.prompts.templates import get_filter_prompt


# Set up logging
logger = logging.getLogger(__name__)


def _extract_text_from_message(message) -> str:
    """
    Extract text from AIMessage, handling both string and content block formats.

    With output_version="responses/v1", AIMessage.content may be a list of
    content blocks instead of a plain string.
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


def _extract_json_from_response(raw_results: str) -> str | None:
    """
    Extract JSON from LLM response that may contain reasoning text.

    Handles cases where JSON is:
    1. Wrapped in ```json ... ``` markdown blocks anywhere in the text
    2. Raw JSON object embedded in text
    3. Plain JSON response
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


def _parse_filter_results(raw_results: str | None) -> dict[str, Any] | None:
    """Parse filter results from string to dict."""
    if not raw_results:
        return None

    try:
        # Extract JSON from response (handles reasoning text + markdown)
        json_content = _extract_json_from_response(raw_results)

        if not json_content:
            logger.warning("Could not extract JSON from filter response")
            logger.debug(f"Raw response (first 500 chars): {raw_results[:500]}")
            return None

        parsed = json.loads(json_content)

        # Validate structure
        if isinstance(parsed, dict) and "urls" in parsed:
            urls = parsed.get("urls", [])
            logger.info(f"Successfully parsed filter results: {len(urls)} URLs")
            return {
                "urls": urls,
                "total_urls": parsed.get("total_urls", len(urls)),
            }

        logger.warning(f"Parsed JSON missing 'urls' key. Keys found: {list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in filter results: {e}")
        logger.debug(f"Raw response (first 500 chars): {raw_results[:500] if raw_results else 'None'}")
        return None


@traceable(name="filter_node")
async def filter_node(state: ProductResearchState) -> dict:
    """
    Filter search results to keep only relevant product URLs.

    This node:
    1. Takes the search results from the previous search node
    2. Uses an LLM to analyze and filter URLs based on relevance
    3. Returns the filtered URLs list

    Args:
        state: Current workflow state with search results

    Returns:
        Dict with filtered_urls and total_filtered_urls
    """
    # Check if we have search results
    search_results = state.get("current_search_results")
    search_successful = state.get("search_successful", False)

    if not search_successful or not search_results:
        # No results to filter
        return {
            "filtered_urls": [],
            "total_filtered_urls": 0,
        }

    # Get product info
    barcode = state.get("barcode", "")
    sku = state.get("sku", "")
    title = state.get("title", "")

    # Get the filter prompt
    prompt = get_filter_prompt(
        barcode=barcode,
        sku=sku,
        title=title,
        search_results=search_results,
    )

    # Initialize the model with Responses API
    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
        use_responses_api=True,
        output_version="responses/v1",
    )

    try:
        # Execute filter
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Please filter the search results and return the relevant URLs."),
        ]

        response = await model.ainvoke(messages)
        raw_result = _extract_text_from_message(response)

        # Parse the results
        parsed = _parse_filter_results(raw_result)

        if parsed:
            urls = parsed.get("urls", [])
            return {
                "filtered_urls": urls,
                "total_filtered_urls": len(urls),
            }

        # Parsing failed, return empty
        return {
            "filtered_urls": [],
            "total_filtered_urls": 0,
        }

    except Exception as e:
        print(f"Filter error: {e}")
        return {
            "filtered_urls": [],
            "total_filtered_urls": 0,
        }
