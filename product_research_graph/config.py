"""Search configuration definitions for the LangGraph workflow."""

from dataclasses import dataclass
from typing import Literal

from product_research_graph.prompts.templates import SearchPromptKey


ToolType = Literal["google_mcp", "yahoo_mcp", "openai_web_search"]


@dataclass
class SearchConfig:
    """Configuration for a single search attempt."""

    name: str  # Unique identifier for this config
    tool_type: ToolType  # Which tool to use
    prompt_key: SearchPromptKey  # Which prompt template to use
    input_template: str  # Template for generating search input

    def to_dict(self) -> dict:
        """Convert to dictionary for state storage."""
        return {
            "name": self.name,
            "tool_type": self.tool_type,
            "prompt_key": self.prompt_key,
            "input_template": self.input_template,
        }


# ============================================================================
# BARCODE SEARCH CONFIGURATIONS
# ============================================================================

BARCODE_SEARCH_CONFIGS = [
    SearchConfig(
        name="barcode_google",
        tool_type="google_mcp",
        prompt_key="barcode_google",
        input_template="Barcode: {barcode}",
    ),
    SearchConfig(
        name="barcode_yahoo",
        tool_type="yahoo_mcp",
        prompt_key="barcode_yahoo",
        input_template="Barcode: {barcode}",
    ),
    SearchConfig(
        name="barcode_openai",
        tool_type="openai_web_search",
        prompt_key="barcode_openai",
        input_template="Barcode: {barcode}",
    ),
]


# ============================================================================
# SKU SEARCH CONFIGURATIONS
# ============================================================================

SKU_SEARCH_CONFIGS = [
    SearchConfig(
        name="sku_google",
        tool_type="google_mcp",
        prompt_key="sku_google",
        input_template="SKU: {sku}",
    ),
    SearchConfig(
        name="sku_yahoo",
        tool_type="yahoo_mcp",
        prompt_key="sku_yahoo",
        input_template="SKU: {sku}",
    ),
    SearchConfig(
        name="sku_openai",
        tool_type="openai_web_search",
        prompt_key="sku_openai",
        input_template="SKU: {sku}",
    ),
]


# ============================================================================
# TITLE + SKU SEARCH CONFIGURATIONS
# ============================================================================

TITLE_SKU_SEARCH_CONFIGS = [
    SearchConfig(
        name="title_sku_google",
        tool_type="google_mcp",
        prompt_key="title_sku_google",
        input_template="Title: {title}, SKU: {sku}",
    ),
    SearchConfig(
        name="all_fields_openai",
        tool_type="openai_web_search",
        prompt_key="all_fields_openai",
        input_template="This is the product: Barcode/UPC: {barcode}, Product SKU/part number: {sku}, Title: {title}",
    ),
]


# ============================================================================
# COMBINED CONFIGURATIONS
# ============================================================================

# All search configurations (used when barcode is available)
ALL_SEARCH_CONFIGS = BARCODE_SEARCH_CONFIGS + SKU_SEARCH_CONFIGS + TITLE_SKU_SEARCH_CONFIGS

# SKU-only configurations (used when barcode is not available)
SKU_ONLY_SEARCH_CONFIGS = SKU_SEARCH_CONFIGS + TITLE_SKU_SEARCH_CONFIGS


def get_search_configs(has_barcode: bool) -> list[SearchConfig]:
    """
    Get the appropriate search configurations based on available data.

    Args:
        has_barcode: Whether the product has a barcode

    Returns:
        List of SearchConfig objects to try in order
    """
    if has_barcode:
        return ALL_SEARCH_CONFIGS
    return SKU_ONLY_SEARCH_CONFIGS


def get_search_configs_as_dicts(has_barcode: bool) -> list[dict]:
    """
    Get search configurations as dictionaries for state storage.

    Args:
        has_barcode: Whether the product has a barcode

    Returns:
        List of config dictionaries
    """
    configs = get_search_configs(has_barcode)
    return [config.to_dict() for config in configs]


# Tool name mapping for prompts
TOOL_NAMES = {
    "google_mcp": "Google Search tool",
    "yahoo_mcp": "Yahoo Search tool",
    "openai_web_search": "Web Search tool",
}


# ============================================================================
# SEARCH CONFIG TO NODE MAPPING
# ============================================================================

# Maps search config names to their corresponding dedicated node names
# Used by the search_dispatcher to route to the correct node
SEARCH_CONFIG_TO_NODE: dict[str, str] = {
    "barcode_google": "search_barcode_google",
    "barcode_yahoo": "search_barcode_yahoo",
    "barcode_openai": "search_barcode_openai",
    "sku_google": "search_sku_google",
    "sku_yahoo": "search_sku_yahoo",
    "sku_openai": "search_sku_openai",
    "title_sku_google": "search_title_sku_google",
    "all_fields_openai": "search_all_fields_openai",
}


def get_tool_display_name(tool_type: ToolType) -> str:
    """Get the display name for a tool type."""
    return TOOL_NAMES.get(tool_type, "search tool")
