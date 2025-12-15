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
    # Prefer cleaned data if available (from image_urls_cleanup node)
    cleaned_pages = state.get("cleaned_validated_pages")
    validated_pages = cleaned_pages if cleaned_pages else state.get("validated_pages", [])

    cleaned_images_count = state.get("cleaned_total_validated_images")
    total_validated_images = (
        cleaned_images_count
        if cleaned_images_count is not None
        else state.get("total_validated_images", 0)
    )

    invalid_urls = state.get("invalid_urls", [])
    total_checked = state.get("total_checked", 0)

    # If total_validated_images wasn't set correctly, calculate it
    if total_validated_images == 0 and validated_pages:
        total_validated_images = sum(
            len(page.get("image_urls", [])) for page in validated_pages
        )

    # Convert validated_pages to the expected format
    formatted_pages = []
    for page in validated_pages:
        # Get weight data with defaults
        weight = page.get("weight", {})
        weight_formatted = {
            "unit_of_measure": weight.get("unit_of_measure", "") if weight else "",
            "value": weight.get("value") if weight else None,
        }

        # Get dimensions data with defaults
        dimensions = page.get("product_dimensions", {})
        dimensions_formatted = {
            "length": dimensions.get("length") if dimensions else None,
            "width": dimensions.get("width") if dimensions else None,
            "height": dimensions.get("height") if dimensions else None,
        }

        formatted_pages.append({
            "url": page.get("url", ""),
            "validation_method": page.get("validation_method", "unknown"),
            "image_urls": page.get("image_urls", []),
            "reasoning": page.get("reasoning", ""),
            "product_description": page.get("product_description", ""),
            "brand": page.get("brand", ""),
            "weight": weight_formatted,
            "product_dimensions": dimensions_formatted,
            "is_shopify": page.get("is_shopify"),
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
