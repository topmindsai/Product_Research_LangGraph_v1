"""Configuration and environment settings."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API key validation
serp_api_key = os.environ.get("BA_SERPAPI_KEY")
if not serp_api_key:
    raise RuntimeError("BA_SERPAPI_KEY is not set – MCP auth header would be empty.")

# OpenAI API key (required for LangGraph implementation)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set – required for LangGraph implementation.")


def init_tracing():
    """Initialize LangSmith tracing for LangGraph."""
    # Set these in your .env file:
    # LANGCHAIN_TRACING_V2=true
    # LANGCHAIN_PROJECT=Product_Research_Agent_v3
    # LANGCHAIN_API_KEY=your_langsmith_api_key
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "Product_Research_Agent_v3")


# Initialize tracing on module load
init_tracing()


# LangGraph-specific configuration
class LangGraphConfig:
    """Configuration for LangGraph implementation."""

    # Model settings
    SEARCH_MODEL = os.getenv("LANGGRAPH_SEARCH_MODEL", "gpt-5-mini")
    # Validation model - supports OpenAI (gpt-*) and Anthropic (claude-*) models
    # Examples: "gpt-5.1", "claude-sonnet-4-5-20250929"
    VALIDATION_MODEL = os.getenv("LANGGRAPH_VALIDATION_MODEL", "gpt-5.1")

    # Retry settings
    MAX_SEARCH_RETRIES = int(os.getenv("LANGGRAPH_MAX_SEARCH_RETRIES", "3"))
    MAX_SCRAPE_RETRIES = int(os.getenv("LANGGRAPH_MAX_SCRAPE_RETRIES", "3"))

    # MCP Server URLs
    SERP_MCP_URL = os.getenv(
        "SERP_MCP_URL",
        "https://serp-mcp-ts-v1-production.up.railway.app/mcp"
    )
    ZYTE_MCP_URL = os.getenv(
        "ZYTE_MCP_URL",
        "https://zytemcp-production.up.railway.app/mcp"
    )

    # Timeout settings (in seconds)
    MCP_TIMEOUT = float(os.getenv("MCP_TIMEOUT", "30.0"))
    MCP_SSE_TIMEOUT = float(os.getenv("MCP_SSE_TIMEOUT", "120.0"))

    # Graph execution settings
    # Default of 25 is insufficient for 8 search configs (26 steps needed)
    RECURSION_LIMIT = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "50"))
