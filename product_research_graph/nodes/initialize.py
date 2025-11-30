"""Initialize node for the Product Research workflow."""

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.config import get_search_configs_as_dicts


@traceable(name="initialize_node")
def initialize_node(state: ProductResearchState) -> dict:
    """
    Initialize the workflow state based on input product data.

    This node:
    1. Determines which search configurations to use based on barcode availability
    2. Sets the search type label for output
    3. Initializes all state counters to zero

    Args:
        state: Current workflow state with barcode, sku, title

    Returns:
        Dict with updated state fields
    """
    barcode = state.get("barcode", "")
    sku = state.get("sku", "")
    title = state.get("title", "")

    # Determine if we have a barcode
    has_barcode = bool(barcode and barcode.strip())

    # Get appropriate search configurations
    search_configs = get_search_configs_as_dicts(has_barcode)

    # Determine search type label for output
    search_type_label = "barcode" if has_barcode else "sku"

    return {
        "search_configs": search_configs,
        "search_index": 0,
        "search_type_label": search_type_label,
        # Reset all state
        "current_search_results": None,
        "retry_count": 0,
        "search_successful": False,
        "filtered_urls": [],
        "total_filtered_urls": 0,
        "total_validated_images": 0,
        "validated_pages": [],
        "invalid_urls": [],
        "total_checked": 0,
        "final_result": None,
    }
