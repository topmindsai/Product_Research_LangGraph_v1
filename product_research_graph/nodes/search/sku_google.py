"""Search node for SKU search using Google MCP."""

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_search


@traceable(name="search_sku_google")
async def search_sku_google_node(state: ProductResearchState) -> dict:
    """
    Search for product using SKU value via Google MCP.

    This node executes an SKU-based search using the Google Search MCP tool.
    It uses the SKU/part number as the sole search query.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including search results
    """
    return await execute_search(
        state=state,
        tool_type="google_mcp",
        prompt_key="sku_google",
        input_template="SKU: {sku}",
        node_name="search_sku_google",
    )
