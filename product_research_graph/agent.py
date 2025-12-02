"""LangGraph StateGraph definition for the Product Research workflow."""

from typing import Literal

from langgraph.graph import StateGraph, START, END

from product_research_graph.state import (
    ProductResearchState,
    ProductResearchInputState,
    ProductResearchOutputState,
)
from product_research_graph.nodes import (
    initialize_node,
    filter_node,
    validate_node,
    finalize_node,
)
from product_research_graph.nodes.search import (
    search_dispatcher,
    search_barcode_google_node,
    search_barcode_yahoo_node,
    search_barcode_openai_node,
    search_sku_google_node,
    search_sku_yahoo_node,
    search_sku_openai_node,
    search_title_sku_google_node,
    search_all_fields_openai_node,
)


# List of all search node names for edge definitions
SEARCH_NODE_NAMES = [
    "search_barcode_google",
    "search_barcode_yahoo",
    "search_barcode_openai",
    "search_sku_google",
    "search_sku_yahoo",
    "search_sku_openai",
    "search_title_sku_google",
    "search_all_fields_openai",
]


def should_continue_search(state: ProductResearchState) -> Literal["continue", "done"]:
    """
    Determine if we should continue searching or finalize.

    Continue if:
    - We haven't found any validated images yet (total_validated_images < 1)
    - AND we haven't exhausted all search configurations

    Stop (finalize) if:
    - We have found at least one validated image
    - OR we have tried all search configurations

    Args:
        state: Current workflow state

    Returns:
        "continue" to try the next search config, "done" to finalize
    """
    total_validated_images = state.get("total_validated_images", 0)
    search_index = state.get("search_index", 0)
    search_configs = state.get("search_configs", [])

    # Stop if we found images
    if total_validated_images >= 1:
        return "done"

    # Stop if we've exhausted all search configs
    if search_index >= len(search_configs):
        return "done"

    # Continue to next search config
    return "continue"


def create_product_research_graph() -> StateGraph:
    """
    Create and compile the Product Research StateGraph.

    The graph structure is:

    START
      │
      ▼
    initialize
      │
      ▼
    search_dispatcher ◄───────────────────────────────────┐
      │ (routes based on search_index)                    │
      ├─► search_barcode_google ──┐                       │
      ├─► search_barcode_yahoo ───┤                       │
      ├─► search_barcode_openai ──┤                       │
      ├─► search_sku_google ──────┤                       │
      ├─► search_sku_yahoo ───────┼──► filter ──► validate ──► should_continue?
      ├─► search_sku_openai ──────┤                       │         │
      ├─► search_title_sku_google ┤                       │    (continue)
      └─► search_all_fields_openai┘                       │         │
                                                          └─────────┘
                                                                │
                                                              (done)
                                                                │
                                                                ▼
                                                            finalize
                                                                │
                                                                ▼
                                                               END

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph with explicit input/output schemas
    # This ensures LangSmith only shows barcode, sku, title as required inputs
    workflow = StateGraph(
        ProductResearchState,
        input=ProductResearchInputState,
        output=ProductResearchOutputState,
    )

    # Add nodes - core workflow nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("filter", filter_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("finalize", finalize_node)

    # Add nodes - search dispatcher (uses Command for routing)
    workflow.add_node("search_dispatcher", search_dispatcher)

    # Add nodes - dedicated search nodes
    workflow.add_node("search_barcode_google", search_barcode_google_node)
    workflow.add_node("search_barcode_yahoo", search_barcode_yahoo_node)
    workflow.add_node("search_barcode_openai", search_barcode_openai_node)
    workflow.add_node("search_sku_google", search_sku_google_node)
    workflow.add_node("search_sku_yahoo", search_sku_yahoo_node)
    workflow.add_node("search_sku_openai", search_sku_openai_node)
    workflow.add_node("search_title_sku_google", search_title_sku_google_node)
    workflow.add_node("search_all_fields_openai", search_all_fields_openai_node)

    # Add edges - start to initialize
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "search_dispatcher")

    # Note: search_dispatcher uses Command to route to search nodes,
    # so we don't need explicit edges from dispatcher to search nodes

    # Add edges - all search nodes converge to filter
    for search_node in SEARCH_NODE_NAMES:
        workflow.add_edge(search_node, "filter")

    # Add edges - filter to validate
    workflow.add_edge("filter", "validate")

    # Add conditional edge for the search loop
    workflow.add_conditional_edges(
        "validate",
        should_continue_search,
        {
            "continue": "search_dispatcher",  # Go back to dispatcher for next search
            "done": "finalize",               # We're done searching
        },
    )

    # Final edge
    workflow.add_edge("finalize", END)

    return workflow


def get_compiled_graph():
    """
    Get a compiled graph ready for execution.

    Includes default recursion_limit for LangSmith Studio/Cloud deployments.

    Returns:
        Compiled graph with invoke/ainvoke/stream/astream methods
    """
    from product_research.config.settings import LangGraphConfig

    workflow = create_product_research_graph()
    compiled = workflow.compile()
    # Bake in default recursion_limit for Studio/Cloud deployments
    return compiled.with_config(recursion_limit=LangGraphConfig.RECURSION_LIMIT)


# Pre-compile the graph for reuse
_compiled_graph = None


def get_graph():
    """
    Get the compiled graph, creating it if necessary.

    This uses a singleton pattern to avoid recompiling the graph
    on every request.

    Returns:
        Compiled graph instance
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = get_compiled_graph()
    return _compiled_graph
