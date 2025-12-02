"""Filter node for the Product Research workflow."""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.prompts.templates import get_filter_prompt
from product_research_graph.utils.parsing import (
    extract_text_from_message,
    extract_json_from_response,
)


# Set up logging
logger = logging.getLogger(__name__)


def _parse_filter_results(raw_results: str | None) -> dict[str, Any] | None:
    """Parse filter results from string to dict."""
    if not raw_results:
        return None

    try:
        # Extract JSON from response (handles reasoning text + markdown)
        json_content = extract_json_from_response(raw_results)

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
        raw_result = extract_text_from_message(response)

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
