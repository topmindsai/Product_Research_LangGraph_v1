"""Search node for barcode search using Yahoo MCP."""

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_search


@traceable(name="search_barcode_yahoo")
async def search_barcode_yahoo_node(state: ProductResearchState) -> dict:
    """
    Search for product using barcode value via Yahoo MCP.

    This node executes a barcode-based search using the Yahoo Search MCP tool.
    It uses the barcode value as the sole search query.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including search results
    """
    return await execute_search(
        state=state,
        tool_type="yahoo_mcp",
        prompt_key="barcode_yahoo",
        input_template="Barcode: {barcode}",
        node_name="search_barcode_yahoo",
    )
