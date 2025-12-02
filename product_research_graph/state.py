"""State schema definition for the Product Research workflow."""

from typing import TypedDict, Annotated, Any, NotRequired
from operator import add


def merge_lists(left: list, right: list) -> list:
    """Reducer that merges two lists, avoiding duplicates for dicts."""
    if not left:
        return right
    if not right:
        return left
    # For simple values, just concatenate
    return left + right


class ProductResearchInputState(TypedDict):
    """Input schema - only the fields required to start the workflow."""
    barcode: NotRequired[str]  # Optional - can be empty
    sku: str
    title: str


class ProductResearchOutputState(TypedDict):
    """Output schema - only the final result."""
    final_result: dict[str, Any] | None


class SearchConfigDict(TypedDict):
    """Configuration for a single search attempt."""
    name: str
    tool_type: str  # "google_mcp", "yahoo_mcp", "openai_web_search"
    input_template: str
    prompt_key: str  # Key to look up prompt template


class ValidatedPageDict(TypedDict):
    """A validated product page with extracted images."""
    url: str
    validation_method: str  # "barcode" or "sku"
    image_urls: list[str]


class ProductResearchState(TypedDict):
    """
    State schema for the Product Research workflow.

    This state flows through all nodes in the graph:
    initialize -> search -> filter -> validate -> (loop or finalize)
    """
    # ===== Input (set once at start) =====
    barcode: str
    sku: str
    title: str

    # ===== Search Configuration =====
    search_configs: list[SearchConfigDict]  # List of search configurations to try
    search_index: int                        # Current index in search_configs
    search_type_label: str                   # "barcode" or "sku" for output

    # ===== Search State =====
    current_search_results: str | None  # JSON string of search results
    retry_count: int                     # Current retry count (0-3)
    search_successful: bool              # Whether current search produced results

    # ===== Filter State =====
    filtered_urls: list[str]       # URLs that passed filtering
    total_filtered_urls: int       # Count of filtered URLs

    # ===== Validation State =====
    total_validated_images: Annotated[int, add]              # Total images found (accumulates)
    validated_pages: Annotated[list[ValidatedPageDict], merge_lists]  # Accumulates
    invalid_urls: Annotated[list[str], merge_lists]          # Accumulates
    total_checked: Annotated[int, add]                       # URLs checked so far (accumulates)

    # ===== Final Output =====
    final_result: dict[str, Any] | None  # Final output matching schema


# Initial state factory
def create_initial_state(barcode: str, sku: str, title: str) -> ProductResearchState:
    """Create initial state for a product research workflow."""
    return ProductResearchState(
        # Input
        barcode=barcode,
        sku=sku,
        title=title,
        # Search configuration (populated by initialize node)
        search_configs=[],
        search_index=0,
        search_type_label="",
        # Search state
        current_search_results=None,
        retry_count=0,
        search_successful=False,
        # Filter state
        filtered_urls=[],
        total_filtered_urls=0,
        # Validation state
        total_validated_images=0,
        validated_pages=[],
        invalid_urls=[],
        total_checked=0,
        # Final output
        final_result=None,
    )
