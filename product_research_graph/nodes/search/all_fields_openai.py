"""Search node for all-fields search using OpenAI web search.

This node routes directly to finalize (skips filter/validate).
It extracts source_url and image_urls from the LLM's structured response
(which performs its own web browsing and image extraction).
"""

import json
import logging

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_search


logger = logging.getLogger(__name__)


@traceable(name="search_all_fields_openai")
async def search_all_fields_openai_node(state: ProductResearchState) -> dict:
    """
    Search for product using all available fields via OpenAI web search.

    This node executes a comprehensive search using OpenAI's direct web search
    capability (no MCP tool). It uses barcode, SKU, and title together for
    the most thorough search possible.

    This is typically the last search configuration tried, used when more
    targeted searches (barcode-only, SKU-only) have failed.

    Unlike other search nodes, this one routes directly to finalize (skips
    filter/validate). The LLM performs its own web browsing and image extraction,
    returning structured data with source_url and image_urls fields.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including validated_pages for finalize
    """
    # Execute the base search
    base_result = await execute_search(
        state=state,
        tool_type="openai_web_search",
        prompt_key="all_fields_openai",
        input_template="This is the product: Barcode/UPC: {barcode}, Product SKU/part number: {sku}, Title: {title}",
        node_name="search_all_fields_openai",
    )

    # Extract URLs and images from search results and convert to validated_pages format
    # The SEARCH_ALL_FIELDS_TEMPLATE prompt returns: {"items": [{"source_url": "...", "image_urls": [...]}]}
    validated_pages = []
    urls_found = []
    total_images = 0

    search_results_json = base_result.get("current_search_results")
    if search_results_json:
        try:
            parsed = json.loads(search_results_json)
            # Check "items" first (expected from SEARCH_ALL_FIELDS_TEMPLATE), then "results" as fallback
            results = parsed.get("items", parsed.get("results", []))

            for item in results:
                # Extract URL - "source_url" is the expected field from SEARCH_ALL_FIELDS_TEMPLATE
                # Fallback to "url" or "link" for robustness
                url = item.get("source_url") or item.get("url") or item.get("link")

                # Extract image URLs - already provided by the LLM's web browsing
                image_urls = item.get("image_urls", [])

                if url:
                    urls_found.append(url)
                    validated_pages.append({
                        "url": url,
                        "validation_method": "all_fields_search",
                        "image_urls": image_urls,
                        "reasoning": "Validated via comprehensive all-fields OpenAI web search",
                        "product_description": item.get("product_description", ""),
                    })
                    total_images += len(image_urls)

            logger.info(
                f"[search_all_fields_openai] Extracted {len(urls_found)} URLs "
                f"with {total_images} images from search results"
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[search_all_fields_openai] Failed to parse search results: {e}")

    # Return all fields needed by finalize
    # Since this node skips filter/validate, we must provide the fields finalize expects
    return {
        # Standard search fields
        "current_search_results": base_result.get("current_search_results"),
        "search_successful": base_result.get("search_successful", False),
        "retry_count": 0,
        "search_index": base_result.get("search_index", state.get("search_index", 0) + 1),

        # Fields normally set by filter/validate (required by finalize)
        # These use reducers (add/merge_lists) so they'll accumulate with previous iterations
        "validated_pages": validated_pages,
        "invalid_urls": [],
        "total_checked": len(urls_found),
        "total_validated_images": total_images,
    }
