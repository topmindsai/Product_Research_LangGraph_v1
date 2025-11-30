"""Search node for title + SKU search using Google MCP."""

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_search


@traceable(name="search_title_sku_google")
async def search_title_sku_google_node(state: ProductResearchState) -> dict:
    """
    Search for product using title and SKU combination via Google MCP.

    This node executes a combined title+SKU search using the Google Search MCP tool.
    It combines the product title and SKU/part number for a more comprehensive search.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including search results
    """
    return await execute_search(
        state=state,
        tool_type="google_mcp",
        prompt_key="title_sku_google",
        input_template="Title: {title}, SKU: {sku}",
        node_name="search_title_sku_google",
    )
