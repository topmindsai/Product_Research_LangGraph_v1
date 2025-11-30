"""MCP tool adapters using langchain-mcp-adapters.

This module provides MCP tool connections using the stateless pattern.
Uses Streamable HTTP transport for the MCP servers.

IMPORTANT: Uses client.get_tools() directly (not session context manager)
to avoid the session closing before tools can be invoked.
"""

import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from product_research.config.settings import serp_api_key, LangGraphConfig


# Set up logging
logger = logging.getLogger(__name__)

# MCP Server URLs (from settings for consistency)
SERP_MCP_URL = LangGraphConfig.SERP_MCP_URL
ZYTE_MCP_URL = LangGraphConfig.ZYTE_MCP_URL

# Exact tool names from MCP servers (must match server definitions)
GOOGLE_SEARCH_TOOL_NAME = "Google_Search"
YAHOO_SEARCH_TOOL_NAME = "Yahoo_Search"
ZYTE_SCRAPE_TOOL_NAME = "scrape_product_optimized"


async def get_serp_tools() -> list[BaseTool]:
    """
    Get all search tools from the SERP MCP server.

    Uses client.get_tools() directly which manages connections internally.
    This is the stateless approach recommended in langchain-mcp-adapters docs.

    Returns:
        List of BaseTool instances from the SERP MCP server.
    """
    client = MultiServerMCPClient({
        "serp_mcp": {
            "url": SERP_MCP_URL,
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {serp_api_key}"
            },
            "timeout": LangGraphConfig.MCP_TIMEOUT,
        }
    })

    try:
        tools = await client.get_tools()
        logger.info(f"Loaded {len(tools)} tools from SERP MCP: {[t.name for t in tools]}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load SERP MCP tools: {e}")
        return []


async def get_zyte_tools() -> list[BaseTool]:
    """
    Get all tools from the Zyte MCP server.

    Uses client.get_tools() directly which manages connections internally.
    This is the stateless approach recommended in langchain-mcp-adapters docs.

    Returns:
        List of BaseTool instances from the Zyte MCP server.
    """
    client = MultiServerMCPClient({
        "zyte_mcp": {
            "url": ZYTE_MCP_URL,
            "transport": "streamable_http",
            "timeout": 60.0,
        }
    })

    try:
        tools = await client.get_tools()
        logger.info(f"Loaded {len(tools)} tools from Zyte MCP: {[t.name for t in tools]}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load Zyte MCP tools: {e}")
        return []


async def get_google_search_tool() -> BaseTool | None:
    """
    Get the Google Search tool from the SERP MCP server.

    Returns:
        The Google_Search tool or None if not found.
    """
    tools = await get_serp_tools()

    for tool in tools:
        if tool.name == GOOGLE_SEARCH_TOOL_NAME:
            logger.debug(f"Found Google Search tool: {tool.name}")
            return tool

    # Fallback: try case-insensitive match
    for tool in tools:
        if "google" in tool.name.lower() and "search" in tool.name.lower():
            logger.warning(f"Using fallback match for Google Search: {tool.name}")
            return tool

    logger.error(f"Google Search tool not found. Available tools: {[t.name for t in tools]}")
    return None


async def get_yahoo_search_tool() -> BaseTool | None:
    """
    Get the Yahoo Search tool from the SERP MCP server.

    Returns:
        The Yahoo_Search tool or None if not found.
    """
    tools = await get_serp_tools()

    for tool in tools:
        if tool.name == YAHOO_SEARCH_TOOL_NAME:
            logger.debug(f"Found Yahoo Search tool: {tool.name}")
            return tool

    # Fallback: try case-insensitive match
    for tool in tools:
        if "yahoo" in tool.name.lower() and "search" in tool.name.lower():
            logger.warning(f"Using fallback match for Yahoo Search: {tool.name}")
            return tool

    logger.error(f"Yahoo Search tool not found. Available tools: {[t.name for t in tools]}")
    return None


async def get_zyte_scrape_tool() -> BaseTool | None:
    """
    Get the Zyte scraping tool from the Zyte MCP server.

    Returns:
        The scrape_product_optimized tool or None if not found.
    """
    tools = await get_zyte_tools()

    for tool in tools:
        if tool.name == ZYTE_SCRAPE_TOOL_NAME:
            logger.debug(f"Found Zyte scrape tool: {tool.name}")
            return tool

    # Fallback: try partial match
    for tool in tools:
        if "scrape" in tool.name.lower():
            logger.warning(f"Using fallback match for Zyte scrape: {tool.name}")
            return tool

    logger.error(f"Zyte scrape tool not found. Available tools: {[t.name for t in tools]}")
    return None


def get_tool_name_for_type(tool_type: str) -> str:
    """
    Get the exact MCP tool name for a given tool type.

    Args:
        tool_type: One of "google_mcp", "yahoo_mcp", "zyte_mcp"

    Returns:
        The exact tool name as defined on the MCP server.
    """
    mapping = {
        "google_mcp": GOOGLE_SEARCH_TOOL_NAME,
        "yahoo_mcp": YAHOO_SEARCH_TOOL_NAME,
        "zyte_mcp": ZYTE_SCRAPE_TOOL_NAME,
    }
    return mapping.get(tool_type, tool_type)
