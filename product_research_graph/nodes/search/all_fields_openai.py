"""Search node for all-fields search using OpenAI web search."""

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_search


@traceable(name="search_all_fields_openai")
async def search_all_fields_openai_node(state: ProductResearchState) -> dict:
    """
    Search for product using all available fields via OpenAI web search.

    This node executes a comprehensive search using OpenAI's direct web search
    capability (no MCP tool). It uses barcode, SKU, and title together for
    the most thorough search possible.

    This is typically the last search configuration tried, used when more
    targeted searches (barcode-only, SKU-only) have failed.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including search results
    """
    return await execute_search(
        state=state,
        tool_type="openai_web_search",
        prompt_key="all_fields_openai",
        input_template="This is the product: Barcode/UPC: {barcode}, Product SKU/part number: {sku}, Title: {title}",
        node_name="search_all_fields_openai",
    )
