# Product Research Agent v2

AI-powered product image search system using LangGraph.

## Overview

This application uses LangGraph to orchestrate a multi-node workflow that searches the web for product images using barcode, SKU, or title identifiers. It validates that found images match the exact product variant and returns high-quality, Shopify-ready image URLs.

## Features

- **LangGraph Workflow**: StateGraph-based orchestration with conditional routing and retry logic
- **Multiple Search Sources**: Google, Yahoo, and OpenAI web search via MCP tools
- **8-Step Search Strategy**: Sequential fallback from barcode → SKU → title+SKU combinations
- **Strict Image Validation**: Ensures images match exact product variants (not similar products)
- **Batch Processing**: Process CSV/Excel files with concurrent execution
- **Dual-Mode Execution**: Run as REST API server or CLI tool
- **Structured Output**: Pydantic-validated JSON responses
- **LangSmith Tracing**: Built-in observability with automatic trace saving

## Project Structure

```
Product_Research_Agent_v2/
├── product_research_graph/     # LangGraph agent (standalone package)
│   ├── __init__.py             # Exports: run_workflow, create_product_research_graph
│   ├── agent.py                # StateGraph definition & compilation
│   ├── workflow.py             # Workflow entry point (run_workflow)
│   ├── state.py                # ProductResearchState TypedDict
│   ├── config.py               # Search configurations
│   ├── tracing.py              # LangSmith trace utilities
│   ├── nodes/                  # Graph nodes
│   │   ├── initialize.py       # Setup search configs
│   │   ├── filter.py           # URL filtering via LLM
│   │   ├── validate.py         # Image validation & extraction
│   │   ├── finalize.py         # Format final output
│   │   └── search/             # 8 search strategy nodes + dispatcher
│   ├── tools/                  # MCP tool wrappers
│   │   └── mcp_tools.py        # Google, Yahoo, OpenAI search tools
│   └── prompts/                # Prompt templates
│       └── templates.py        # Search, filter, validation prompts
├── product_research/           # Application layer
│   ├── __init__.py             # Exports: ProductInput, ValidationImageExtractionAgentSchema
│   ├── config/                 # Environment and settings
│   │   └── settings.py
│   ├── schemas/                # Pydantic models
│   │   └── models.py
│   ├── batch/                  # Batch processing
│   │   └── processor.py
│   └── api/                    # FastAPI application
│       └── routes.py
├── main.py                     # CLI entry point
├── langgraph.json              # LangGraph Cloud deployment config
├── pyproject.toml              # Project dependencies
└── .env                        # Environment variables (not committed)
```

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- SerpAPI key (for Google/Yahoo search)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Product_Research_Agent_v2

# Create virtual environment

source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key
BA_SERPAPI_KEY=your_serpapi_key

# LangSmith Tracing (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Product_Research_Agent
LANGCHAIN_API_KEY=your_langsmith_api_key

# LangGraph Settings (optional)
LANGGRAPH_SEARCH_MODEL=gpt-5-mini      # Model for search nodes
LANGGRAPH_VALIDATION_MODEL=gpt-5.1        # Model for validation
LANGGRAPH_MAX_SEARCH_RETRIES=3           # Retries per search config
LANGGRAPH_RECURSION_LIMIT=50             # Max graph steps
```

### Running the Application

**API Server Mode (default):**
```bash
python main.py
# Server starts at http://localhost:8000
```

**CLI Batch Mode:**
```bash
python main.py --batch input.csv --output results.csv
```

**LangGraph Studio (development):**
```bash
langgraph dev
```

## Usage Examples

### Single Product Search (API)

```bash
curl -X POST http://localhost:8000/api/v1/product-images \
  -H "Content-Type: application/json" \
  -d '{
    "barcode": "032054072245",
    "sku": "SQDR45115",
    "title": "Danielson SQDR45115 Squid Rigged 4.5\" Glow/Blue"
  }'
```

```bash
curl -X POST http://localhost:8000/api/v1/product-images \
  -H "Content-Type: application/json" \
  -d '{
    "barcode": "22677322513",
    "sku": "KSJ280NFZ",
    "title": "Williamson KSJ280NFZ Kensaki Jig"
  }'
```

### Batch Processing (API)

```bash
curl -X POST http://localhost:8000/api/v1/product-images/batch \
  -F "file=@products.csv"
```

### Batch Processing (CLI)

```bash
python main.py --batch tests/3_Batch.csv --output tests/test_results/results.csv
```

**Input file format (CSV):**
```csv
barcode,sku,title
032054072245,DANI-SQDR45115,"Danielson SQDR45115 Squid Rigged 4.5"" Glow/Blue"
032054071835,DANI-SQD752130,"Danielson SQD752130 Squid Bait 7.5"" UV Clear"
```

## Programmatic Usage

```python
import asyncio
from product_research_graph import run_workflow
from product_research import ProductInput

async def main():
    product = ProductInput(
        barcode="032054072245",
        sku="DANI-SQDR45115",
        title="Danielson SQDR45115 Squid Rigged"
    )
    result = await run_workflow(product)
    print(f"Found {result.total_validated_images} images")

    for page in result.validated_pages:
        print(f"  Source: {page.url}")
        for img in page.image_urls:
            print(f"    - {img}")

asyncio.run(main())
```

### Using the Graph Directly

```python
from product_research_graph import create_product_research_graph
from product_research_graph.state import create_initial_state

# Create and compile the graph
graph = create_product_research_graph()

# Create initial state
state = create_initial_state(
    barcode="032054072245",
    sku="DANI-SQDR45115",
    title="Danielson SQDR45115 Squid Rigged"
)

# Run with streaming
async for event in graph.astream(state):
    print(event)
```

## Workflow Architecture

The LangGraph workflow follows this execution flow:

```
START
  │
  ├─► initialize_node (setup search configs based on input)
  │         │
  │         ▼
  │   search_dispatcher ─────────────────────────────┐
  │         │                                        │
  │         ▼                                        │
  │   search_*_node (one of 8 search strategies)     │
  │         │                                        │
  │         ▼                                        │
  │   filter_node (LLM filters relevant URLs)        │
  │         │                                        │
  │         ▼                                        │
  │   validate_node (scrape & extract images)        │
  │         │                                        │
  │         ▼                                        │
  │   should_continue? ──── yes (no images) ─────────┘
  │         │
  │         no (images found or configs exhausted)
  │         │
  │         ▼
  │   finalize_node (format output)
  │         │
  └─────────▼
          END
```

**Search Strategy (8 configurations):**
1. Barcode + Google Search
2. Barcode + Yahoo Search
3. Barcode + OpenAI Web Search
4. SKU + Google Search
5. SKU + Yahoo Search
6. SKU + OpenAI Web Search
7. Title+SKU + Google Search
8. All Fields + OpenAI Web Search

The workflow stops on the first successful image extraction (≥1 validated image) or after exhausting all configurations.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/product-images` | Search for single product images |
| POST | `/api/v1/product-images/batch` | Process multiple products from CSV/Excel |
| GET | `/api/v1/batch-results/{filename}` | Download batch results file |
| GET | `/health` | Health check |

## License

See LICENSE file for details.
