"""Initialize node for the Product Research workflow."""

import logging

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.config import get_search_configs_as_dicts


# Set up logging
logger = logging.getLogger(__name__)


def normalize_barcode(barcode: str | int) -> str:
    """
    Normalize barcode to standard 12-digit UPC format.

    Handles common issues:
    - Converts int to string (handles CSV/JSON parsing inconsistencies)
    - Strips whitespace
    - Pads 11-digit barcodes with leading zero (common truncation)
    - Trims 13-digit GTIN-13/EAN-13 to 12-digit UPC (removes leading 0)
    - Trims 14-digit GTIN-14 to 12-digit UPC (removes leading 2 digits)

    Standard formats:
    - UPC-A: 12 digits (most common in US)
    - EAN-13/GTIN-13: 13 digits (international, often starts with 0 for US products)
    - GTIN-14: 14 digits (case/pallet level, first digit is packaging indicator)

    Args:
        barcode: Raw barcode string or int

    Returns:
        Normalized 12-digit UPC barcode, or original if not normalizable
    """
    if not barcode:
        return ""

    # Convert to string first to handle int inputs (from CSV/JSON parsing)
    barcode_str = str(barcode)

    # Strip whitespace and any non-digit characters
    cleaned = ''.join(c for c in barcode_str.strip() if c.isdigit())

    if not cleaned:
        return barcode_str

    original_len = len(cleaned)

    # Already 12 digits - standard UPC-A
    if original_len == 12:
        return cleaned

    # 11 digits - missing leading zero (common truncation)
    if original_len == 11:
        normalized = '0' + cleaned
        logger.info(f"Barcode normalized: {barcode_str} -> {normalized} (added leading zero)")
        return normalized

    # 13 digits - GTIN-13/EAN-13, trim leading 0 if present (common for US products)
    if original_len == 13:
        if cleaned.startswith('0'):
            normalized = cleaned[1:]  # Remove leading 0
            logger.info(f"Barcode normalized: {barcode_str} -> {normalized} (trimmed EAN-13 to UPC)")
            return normalized
        # Non-zero leading digit - keep as-is, search engines may handle it
        logger.warning(f"Barcode is 13-digit EAN without leading 0: {barcode_str}, using as-is")
        return cleaned

    # 14 digits - GTIN-14, trim first 2 digits (packaging indicator + leading 0)
    if original_len == 14:
        normalized = cleaned[2:]  # Remove packaging indicator and leading 0
        logger.info(f"Barcode normalized: {barcode_str} -> {normalized} (trimmed GTIN-14 to UPC)")
        return normalized

    # Other lengths - return cleaned version, log warning
    if original_len < 11:
        logger.warning(f"Barcode too short ({original_len} digits): {barcode_str}")
    elif original_len > 14:
        logger.warning(f"Barcode too long ({original_len} digits): {barcode_str}")

    return cleaned


@traceable(name="initialize_node")
def initialize_node(state: ProductResearchState) -> dict:
    """
    Initialize the workflow state based on input product data.

    This node:
    1. Normalizes barcode to standard 12-digit UPC format
    2. Determines which search configurations to use based on barcode availability
    3. Sets the search type label for output
    4. Initializes all state counters to zero

    Args:
        state: Current workflow state with barcode, sku, title

    Returns:
        Dict with updated state fields (including normalized barcode)
    """
    barcode = state.get("barcode", "")
    sku = state.get("sku", "")
    title = state.get("title", "")

    # Normalize barcode to standard 12-digit UPC format
    normalized_barcode = normalize_barcode(barcode) if barcode else ""

    # Determine if we have a valid barcode
    has_barcode = bool(normalized_barcode and normalized_barcode.strip())

    # Get appropriate search configurations
    search_configs = get_search_configs_as_dicts(has_barcode)

    # Determine search type label for output
    search_type_label = "barcode" if has_barcode else "sku"

    return {
        # Return normalized barcode for use in searches
        "barcode": normalized_barcode,
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
