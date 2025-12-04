"""Search node for all-fields search using OpenAI web search with structured output.

This node uses OpenAI's native web_search_preview tool combined with
structured output (response_format) in a single LLM call for:
- Built-in web search execution (server-side)
- Guaranteed JSON schema compliance

Routes directly to finalize (skips filter/validate) since the LLM
performs its own validation and image extraction.
"""

import logging

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_openai_search_structured
from product_research_graph.prompts.templates import get_search_prompt
from product_research_graph.config import get_tool_display_name


logger = logging.getLogger(__name__)


@traceable(name="search_all_fields_openai")
async def search_all_fields_openai_node(state: ProductResearchState) -> dict:
    """
    Search for product using OpenAI native web_search + structured output.

    This node executes a comprehensive search using OpenAI's built-in
    web_search_preview tool combined with structured output enforcement
    in a single LLM call. No manual JSON parsing required.

    Key features:
    - OpenAI native web_search_preview (server-side execution)
    - OpenAI native structured output (JSON schema enforcement)
    - Both capabilities in a single LLM call
    - Type-safe Pydantic response

    Unlike other search nodes, this one routes directly to finalize (skips
    filter/validate). The LLM performs its own web browsing and image extraction,
    returning structured data guaranteed to match the schema.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including validated_pages for finalize
    """
    search_index = state.get("search_index", 0)

    # Get product info
    barcode = state.get("barcode", "")
    sku = state.get("sku", "")
    title = state.get("title", "")

    # Format the search input
    search_input = f"This is the product: Barcode/UPC: {barcode}, Product SKU/part number: {sku}, Title: {title}"

    # Get the tool display name for the prompt
    tool_display_name = get_tool_display_name("openai_web_search")

    # Get the formatted prompt
    prompt = get_search_prompt(
        prompt_key="all_fields_openai",
        barcode=barcode,
        sku=sku,
        title=title,
        tool_name=tool_display_name,
    )

    logger.info(f"[search_all_fields_openai] Executing with native web_search + structured output")

    # Execute search with OpenAI native web_search + structured output
    # Returns a Pydantic object directly - no JSON parsing needed
    structured_result = await execute_openai_search_structured(
        prompt=prompt,
        query=search_input,
    )

    # Handle failure
    if structured_result is None:
        logger.warning("[search_all_fields_openai] Structured search returned no results")
        return {
            "current_search_results": None,
            "search_successful": False,
            "retry_count": 0,
            "search_index": search_index + 1,
            "validated_pages": [],
            "invalid_urls": [],
            "total_checked": 0,
            "total_validated_images": 0,
        }

    # Convert Pydantic items to validated_pages format
    validated_pages = []
    total_images = 0

    for item in structured_result.items:
        validated_pages.append({
            "url": item.source_url,
            "validation_method": "all_fields_search",
            "image_urls": item.image_urls,
            "reasoning": "Validated via OpenAI native web_search with structured output",
        })
        total_images += len(item.image_urls)

    logger.info(
        f"[search_all_fields_openai] Extracted {len(validated_pages)} URLs "
        f"with {total_images} images from structured response"
    )

    # Return all fields needed by finalize
    # Since this node skips filter/validate, we must provide the fields finalize expects
    return {
        # Standard search fields
        "current_search_results": None,  # Not needed - we use structured response
        "search_successful": len(validated_pages) > 0,
        "retry_count": 0,
        "search_index": search_index + 1,

        # Fields normally set by filter/validate (required by finalize)
        # These use reducers (add/merge_lists) so they'll accumulate with previous iterations
        "validated_pages": validated_pages,
        "invalid_urls": [],
        "total_checked": len(validated_pages),
        "total_validated_images": total_images,
    }
