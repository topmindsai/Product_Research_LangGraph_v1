"""Search node for barcode search using OpenAI web search."""

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_search


@traceable(name="search_barcode_openai")
async def search_barcode_openai_node(state: ProductResearchState) -> dict:
    """
    Search for product using barcode value via OpenAI web search.

    This node executes a barcode-based search using OpenAI's direct web search
    capability (no MCP tool). It uses the barcode value as the sole search query.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including search results
    """
    return await execute_search(
        state=state,
        tool_type="openai_web_search",
        prompt_key="barcode_openai",
        input_template="Barcode: {barcode}",
        node_name="search_barcode_openai",
    )
