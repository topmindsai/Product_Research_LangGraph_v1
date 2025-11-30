"""Search node for barcode search using Google MCP."""

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_search


@traceable(name="search_barcode_google")
async def search_barcode_google_node(state: ProductResearchState) -> dict:
    """
    Search for product using barcode value via Google MCP.

    This node executes a barcode-based search using the Google Search MCP tool.
    It uses the barcode value as the sole search query.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including search results
    """
    return await execute_search(
        state=state,
        tool_type="google_mcp",
        prompt_key="barcode_google",
        input_template="Barcode: {barcode}",
        node_name="search_barcode_google",
    )
