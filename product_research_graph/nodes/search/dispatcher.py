"""Search dispatcher node using LangGraph Command for routing.

This module contains the dispatcher that routes to the appropriate
search node based on the current search_index and configuration.
"""

import logging
from typing import Literal

from langgraph.types import Command

from product_research_graph.state import ProductResearchState
from product_research_graph.config import SEARCH_CONFIG_TO_NODE


# Set up logging
logger = logging.getLogger(__name__)

# Type alias for all possible destination nodes
SearchNodeName = Literal[
    "search_barcode_google",
    "search_barcode_yahoo",
    "search_barcode_openai",
    "search_sku_google",
    "search_sku_yahoo",
    "search_sku_openai",
    "search_title_sku_google",
    "search_all_fields_openai",
    "finalize",
]


def search_dispatcher(
    state: ProductResearchState,
) -> Command[SearchNodeName]:
    """
    Route to the appropriate search node based on the current search configuration.

    This dispatcher reads the search_index from state and uses the SEARCH_CONFIG_TO_NODE
    mapping to determine which dedicated search node should execute next.

    If all search configurations have been exhausted, routes to finalize.

    Args:
        state: Current workflow state containing search_configs and search_index

    Returns:
        Command object that routes to the appropriate search node
    """
    search_index = state.get("search_index", 0)
    search_configs = state.get("search_configs", [])

    # Check if we've exhausted all configs
    if search_index >= len(search_configs):
        logger.info(f"[search_dispatcher] All {len(search_configs)} search configs exhausted, routing to finalize")
        return Command(goto="finalize")

    # Get current config
    current_config = search_configs[search_index]
    config_name = current_config.get("name", "")

    # Look up the node name for this config
    node_name = SEARCH_CONFIG_TO_NODE.get(config_name)

    if not node_name:
        # Fallback: try to construct node name from config name
        logger.warning(f"[search_dispatcher] No mapping found for config '{config_name}', attempting fallback")
        node_name = f"search_{config_name}"

    logger.info(
        f"[search_dispatcher] Routing to {node_name} "
        f"(config {search_index + 1}/{len(search_configs)}: {config_name})"
    )

    return Command(goto=node_name)
