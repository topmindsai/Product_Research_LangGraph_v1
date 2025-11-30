"""Search nodes package for the Product Research workflow.

This package contains:
- search_dispatcher: Routes to the appropriate search node based on config
- 8 dedicated search nodes for different search type + provider combinations
"""

from product_research_graph.nodes.search.dispatcher import search_dispatcher
from product_research_graph.nodes.search.barcode_google import search_barcode_google_node
from product_research_graph.nodes.search.barcode_yahoo import search_barcode_yahoo_node
from product_research_graph.nodes.search.barcode_openai import search_barcode_openai_node
from product_research_graph.nodes.search.sku_google import search_sku_google_node
from product_research_graph.nodes.search.sku_yahoo import search_sku_yahoo_node
from product_research_graph.nodes.search.sku_openai import search_sku_openai_node
from product_research_graph.nodes.search.title_sku_google import search_title_sku_google_node
from product_research_graph.nodes.search.all_fields_openai import search_all_fields_openai_node


__all__ = [
    # Dispatcher
    "search_dispatcher",
    # Barcode search nodes
    "search_barcode_google_node",
    "search_barcode_yahoo_node",
    "search_barcode_openai_node",
    # SKU search nodes
    "search_sku_google_node",
    "search_sku_yahoo_node",
    "search_sku_openai_node",
    # Title + SKU search nodes
    "search_title_sku_google_node",
    # All fields search nodes
    "search_all_fields_openai_node",
]
