"""Search node for SKU search using OpenAI web search."""

from langsmith import traceable

from product_research_graph.state import ProductResearchState
from product_research_graph.nodes.search._base import execute_search


@traceable(name="search_sku_openai")
async def search_sku_openai_node(state: ProductResearchState) -> dict:
    """
    Search for product using SKU value via OpenAI web search.

    This node executes an SKU-based search using OpenAI's direct web search
    capability (no MCP tool). It uses the SKU/part number as the sole search query.

    Args:
        state: Current workflow state

    Returns:
        Dict with updated state fields including search results
    """
    return await execute_search(
        state=state,
        tool_type="openai_web_search",
        prompt_key="sku_openai",
        input_template="SKU: {sku}",
        node_name="search_sku_openai",
    )
