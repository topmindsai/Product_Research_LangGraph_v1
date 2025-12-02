"""Finalize node for the Product Research workflow."""

from typing import Any

from langsmith import traceable

from product_research_graph.state import ProductResearchState


@traceable(name="finalize_node")
def finalize_node(state: ProductResearchState) -> dict:
    """
    Finalize the workflow and prepare the output.

    This node:
    1. Collects all validated pages and images
    2. Constructs the final output in ValidationImageExtractionAgentSchema format
    3. Returns the final result

    Args:
        state: Current workflow state with all accumulated data

    Returns:
        Dict with final_result matching the output schema
    """
    # Get product info
    barcode = state.get("barcode", "")
    sku = state.get("sku", "")
    title = state.get("title", "")
    search_type_label = state.get("search_type_label", "barcode")

    # Get accumulated validation data
    validated_pages = state.get("validated_pages", [])
    invalid_urls = state.get("invalid_urls", [])
    total_checked = state.get("total_checked", 0)
    total_validated_images = state.get("total_validated_images", 0)

    # If total_validated_images wasn't set correctly, calculate it
    if total_validated_images == 0 and validated_pages:
        total_validated_images = sum(
            len(page.get("image_urls", [])) for page in validated_pages
        )

    # Convert validated_pages to the expected format
    formatted_pages = []
    for page in validated_pages:
        formatted_pages.append({
            "url": page.get("url", ""),
            "validation_method": page.get("validation_method", "unknown"),
            "image_urls": page.get("image_urls", []),
            "reasoning": page.get("reasoning", ""),
        })

    # Deduplicate invalid_urls by URL while preserving structure
    seen_invalid_urls: dict[str, dict] = {}
    for item in invalid_urls:
        if isinstance(item, dict):
            url = item.get("url", "")
            if url and url not in seen_invalid_urls:
                seen_invalid_urls[url] = item
        elif isinstance(item, str) and item not in seen_invalid_urls:
            # Backward compatibility for plain strings
            seen_invalid_urls[item] = {"url": item, "reasoning": ""}

    formatted_invalid_urls = list(seen_invalid_urls.values())

    # Construct the final result matching ValidationImageExtractionAgentSchema
    final_result: dict[str, Any] = {
        "product": {
            "barcode": barcode,
            "title": title,
            "sku": sku,
        },
        "search_type": search_type_label,
        "total_checked": total_checked,
        "total_validated_images": total_validated_images,
        "validated_pages": formatted_pages,
        "invalid_urls": formatted_invalid_urls,
    }

    return {
        "final_result": final_result,
    }
