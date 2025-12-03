"""Pydantic models for agent inputs and outputs."""

from pydantic import BaseModel


# Validation/Image Extraction Agent Schemas
class ValidationImageExtractionAgentSchema__Product(BaseModel):
    barcode: str
    title: str
    sku: str


class WeightSchema(BaseModel):
    """Product weight with unit of measure."""
    unit_of_measure: str = ""  # e.g., "lb", "oz", "kg", "g"
    value: float | None = None  # Weight value, None if not found


class ProductDimensionsSchema(BaseModel):
    """Product dimensions in inches."""
    length: float | None = None  # Length in inches, None if not found
    width: float | None = None   # Width in inches, None if not found
    height: float | None = None  # Height in inches, None if not found


class ValidationImageExtractionAgentSchema__ValidatedPagesItem(BaseModel):
    url: str
    validation_method: str
    image_urls: list[str]
    reasoning: str = ""  # Explanation of why this page was validated
    product_description: str = ""  # Product description extracted from the page
    brand: str = ""  # Product brand name extracted from the page
    weight: WeightSchema = WeightSchema()  # Product weight with unit of measure
    product_dimensions: ProductDimensionsSchema = ProductDimensionsSchema()  # Dimensions in inches


class ValidationImageExtractionAgentSchema__InvalidUrlItem(BaseModel):
    """An invalid URL with reasoning for why it was invalidated."""
    url: str
    reasoning: str = ""  # Explanation of why validation failed


class ValidationImageExtractionAgentSchema(BaseModel):
    product: ValidationImageExtractionAgentSchema__Product
    search_type: str
    total_checked: float
    total_validated_images: float
    validated_pages: list[ValidationImageExtractionAgentSchema__ValidatedPagesItem]
    invalid_urls: list[ValidationImageExtractionAgentSchema__InvalidUrlItem]


# Filter Agent Schema
class FilterAgentSchema(BaseModel):
    urls: list[str]
    total_urls: float


# Search Agent Barcode Google Schema
class SearchAgentBarcodeGoogleSchema__ResultsItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchAgentBarcodeGoogleSchema(BaseModel):
    results: list[SearchAgentBarcodeGoogleSchema__ResultsItem]


# Search Agent Barcode Yahoo Schema
class SearchAgentBarcodeYahooSchema__ResultsItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchAgentBarcodeYahooSchema(BaseModel):
    results: list[SearchAgentBarcodeYahooSchema__ResultsItem]


# Search Agent Barcode OpenAI Schema
class SearchAgentBarcodeOpenaiSchema__ResultsItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchAgentBarcodeOpenaiSchema(BaseModel):
    results: list[SearchAgentBarcodeOpenaiSchema__ResultsItem]


# Search Agent SKU Google Schema
class SearchAgentSkuGoogleSchema__ResultsItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchAgentSkuGoogleSchema(BaseModel):
    results: list[SearchAgentSkuGoogleSchema__ResultsItem]


# Search Agent SKU Yahoo Schema
class SearchAgentSkuYahooSchema__ResultsItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchAgentSkuYahooSchema(BaseModel):
    results: list[SearchAgentSkuYahooSchema__ResultsItem]


# Search Agent SKU OpenAI Schema
class SearchAgentSkuOpenaiSchema__ResultsItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchAgentSkuOpenaiSchema(BaseModel):
    results: list[SearchAgentSkuOpenaiSchema__ResultsItem]


# Search Agent Title+SKU Google Schema
class SearchAgentTitleSkuGoogleSchema__ResultsItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchAgentTitleSkuGoogleSchema(BaseModel):
    results: list[SearchAgentTitleSkuGoogleSchema__ResultsItem]


# Search Agent All Fields OpenAI Schema
class SearchAgentAllFieldsOpenaiSchema__ItemsItem(BaseModel):
    source_url: str
    image_urls: list[str]


class SearchAgentAllFieldsOpenaiSchema(BaseModel):
    items: list[SearchAgentAllFieldsOpenaiSchema__ItemsItem]


# API Input/Output Models
class ProductInput(BaseModel):
    """Input model for product image search"""
    barcode: str = ""
    sku: str = ""
    title: str = ""


class BatchResult(BaseModel):
    """Result for a single product in batch processing"""
    barcode: str
    sku: str
    title: str
    result: str  # Full JSON string of ValidationImageExtractionAgentSchema or error


class BatchJobResponse(BaseModel):
    """Response for batch processing"""
    total_products: int
    successful: int
    failed: int
    output_file: str
    results: list[BatchResult]
