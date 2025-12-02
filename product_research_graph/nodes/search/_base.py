"""Shared utilities for search nodes.

This module contains common functions used by all search nodes:
- Text extraction from messages
- JSON parsing from LLM responses
- Tool retrieval helpers
- ReAct agent execution
- Search execution logic
"""

import asyncio
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langsmith import traceable
from langgraph.prebuilt import create_react_agent

from product_research_graph.state import ProductResearchState
from product_research_graph.tools.mcp_tools import (
    get_google_search_tool,
    get_yahoo_search_tool,
)
from product_research_graph.prompts.templates import get_search_prompt
from product_research_graph.config import get_tool_display_name
from product_research_graph.utils.parsing import (
    extract_text_from_message,
    extract_json_from_response,
)


# Set up logging
logger = logging.getLogger(__name__)

# Maximum retries per search config (only for transient errors, not "no results")
MAX_RETRIES = 3


# Phrases that indicate "no results found" (not an error, don't retry)
NO_RESULTS_PHRASES = [
    "no results",
    "hasn't returned any results",
    "hasn't returned any results",  # Different apostrophe
    '"total_results":0',
    '"total_results": 0',
    '"organic_results_state":"fully empty"',
    '"organic_results_state": "fully empty"',
    "your search did not match any documents",
    "did not match any documents",
]


def _is_no_results_response(response: str | None) -> bool:
    """
    Check if the response indicates zero search results (not an error).

    This distinguishes between:
    - "No results found" - Valid response, don't retry
    - API error/timeout - Should retry

    Args:
        response: Raw response string from search

    Returns:
        True if response indicates no results (don't retry), False otherwise
    """
    if not response:
        return False

    lower = response.lower()
    return any(phrase.lower() in lower for phrase in NO_RESULTS_PHRASES)


async def _get_tool_for_type(tool_type: str):
    """Get the appropriate tool based on tool type."""
    logger.info(f"Getting tool for type: {tool_type}")

    if tool_type == "google_mcp":
        tool = await get_google_search_tool()
        if tool:
            logger.info(f"Got Google Search tool: {tool.name}")
        else:
            logger.warning("Google Search tool not available")
        return tool
    elif tool_type == "yahoo_mcp":
        tool = await get_yahoo_search_tool()
        if tool:
            logger.info(f"Got Yahoo Search tool: {tool.name}")
        else:
            logger.warning("Yahoo Search tool not available")
        return tool
    elif tool_type == "openai_web_search":
        # For OpenAI web search, we don't use MCP - the model handles it
        logger.info("OpenAI web search requested - no MCP tool needed")
        return None
    return None


async def _execute_search_with_react_agent(
    tool,
    prompt: str,
    search_input: str,
) -> str | None:
    """
    Execute a search using a ReAct agent for proper tool execution.

    This uses LangGraph's create_react_agent which properly handles:
    - Tool binding to the model
    - Tool call execution
    - Result extraction

    Returns None on failure instead of raising exceptions, allowing
    graceful fallback to the next search config.
    """
    if tool is None:
        logger.error("Cannot execute ReAct agent without a tool")
        return None

    try:
        # Create the model with Responses API
        model = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0,
            use_responses_api=True,
            output_version="responses/v1",
        )

        # Create a ReAct agent with the tool
        agent = create_react_agent(
            model=model,
            tools=[tool],
        )

        # Prepare the input messages
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=search_input),
        ]

        logger.info(f"Invoking ReAct agent with tool: {tool.name}")

        # Invoke the agent with timeout protection
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": messages}),
            timeout=60.0  # 60 second timeout for entire agent execution
        )

        # Extract the final response from the agent
        response_messages = result.get("messages", [])

        # Find the last AI message (the final response)
        final_response = None
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and msg.content:
                # Skip tool call responses (they have tool_calls attribute)
                if not getattr(msg, 'tool_calls', None):
                    final_response = extract_text_from_message(msg)
                    break

        if final_response:
            logger.info(f"ReAct agent returned response ({len(final_response)} chars)")
            return final_response
        else:
            # If no plain AI message, get any content
            for msg in reversed(response_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    return extract_text_from_message(msg)

        logger.warning("ReAct agent returned no usable response")
        return None

    except asyncio.TimeoutError:
        logger.error("ReAct agent execution timed out after 60 seconds")
        return None
    except Exception as e:
        # Log but DON'T re-raise - return None for graceful degradation
        logger.error(f"ReAct agent execution failed: {e}")
        return None


async def _execute_openai_search(
    prompt: str,
    query: str,
) -> str | None:
    """
    Execute a search using OpenAI directly (no tools).

    This is used as a fallback when MCP tools are not available.
    """
    try:
        # Create the model with Responses API and web search tool
        model = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0,
            use_responses_api=True,
            output_version="responses/v1",
        ).bind_tools([{"type": "web_search_preview"}])

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query),
        ]

        response = await model.ainvoke(messages)
        return extract_text_from_message(response)
    except Exception as e:
        logger.error(f"OpenAI search error: {e}")
        return None


def _parse_search_results(raw_results: str | None) -> dict[str, Any] | None:
    """Parse search results from string to dict."""
    if not raw_results:
        return None

    try:
        # Extract JSON from response (handles reasoning text + markdown)
        json_content = extract_json_from_response(raw_results)

        if not json_content:
            logger.warning("Could not extract JSON from search response")
            logger.debug(f"Raw response (first 500 chars): {raw_results[:500]}")
            return None

        parsed = json.loads(json_content)

        # Check if we have results
        if isinstance(parsed, dict):
            results = parsed.get("results", parsed.get("items", []))
            if results:
                logger.info(f"Successfully parsed search results: {len(results)} results")
                return parsed

        logger.warning("Parsed JSON has no results/items")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in search results: {e}")
        logger.debug(f"Raw response (first 500 chars): {raw_results[:500] if raw_results else 'None'}")
        return None


async def execute_search(
    state: ProductResearchState,
    tool_type: str,
    prompt_key: str,
    input_template: str,
    node_name: str,
) -> dict:
    """
    Common search execution logic for all search nodes.

    This function contains the core search logic that all individual
    search nodes share. It handles:
    - Getting the appropriate tool
    - Formatting prompts and inputs
    - Executing searches with retry logic
    - Parsing and returning results

    Args:
        state: Current workflow state
        tool_type: Type of tool to use ("google_mcp", "yahoo_mcp", "openai_web_search")
        prompt_key: Key to look up the prompt template
        input_template: Template for generating search input
        node_name: Name of the search node (for logging)

    Returns:
        Dict with updated state fields including search results
    """
    search_index = state.get("search_index", 0)
    search_configs = state.get("search_configs", [])

    logger.info(f"[{node_name}] Executing search config {search_index + 1}/{len(search_configs)}")

    # Get product info
    barcode = state.get("barcode", "")
    sku = state.get("sku", "")
    title = state.get("title", "")

    # Format the search input
    search_input = input_template.format(
        barcode=barcode,
        sku=sku,
        title=title,
    )

    # Get the tool display name for the prompt
    tool_display_name = get_tool_display_name(tool_type)

    # Get the formatted prompt
    prompt = get_search_prompt(
        prompt_key=prompt_key,
        barcode=barcode,
        sku=sku,
        title=title,
        tool_name=tool_display_name,
    )

    # Execute search with retry logic
    retry_count = 0
    search_result = None

    while retry_count < MAX_RETRIES:
        try:
            if tool_type in ("google_mcp", "yahoo_mcp"):
                # Get MCP tool with proper session management
                tool = await _get_tool_for_type(tool_type)

                if tool:
                    # Use ReAct agent for proper tool execution
                    logger.info(f"[{node_name}] Executing search with ReAct agent (attempt {retry_count + 1}/{MAX_RETRIES})")
                    search_result = await _execute_search_with_react_agent(
                        tool=tool,
                        prompt=prompt,
                        search_input=search_input,
                    )
                else:
                    # Tool not available - this is a critical error now
                    logger.error(f"[{node_name}] MCP tool not available for {tool_type} - cannot proceed")
                    retry_count += 1
                    continue
            else:
                # OpenAI web search - use model directly (fallback)
                logger.info(f"[{node_name}] Using direct OpenAI search (attempt {retry_count + 1}/{MAX_RETRIES})")
                search_result = await _execute_openai_search(prompt, search_input)

            # Check if this is a "no results" response (valid response, don't retry)
            if _is_no_results_response(search_result):
                logger.info(f"[{node_name}] Search returned zero results (not an error) - moving to next config")
                # Don't retry - this is a valid "no results" response
                return {
                    "current_search_results": None,
                    "search_successful": False,
                    "retry_count": 0,
                    "search_index": search_index + 1,
                }

            # Try to parse results
            parsed = _parse_search_results(search_result)
            if parsed:
                logger.info(f"[{node_name}] Search successful with {len(parsed.get('results', []))} results")
                return {
                    "current_search_results": json.dumps(parsed),
                    "search_successful": True,
                    "retry_count": 0,
                    "search_index": search_index + 1,
                }

            # Parsing failed but not a clear "no results" - might be transient, retry
            logger.warning(f"[{node_name}] Search returned unparseable results, retrying... (attempt {retry_count + 1}/{MAX_RETRIES})")
            retry_count += 1

        except Exception as e:
            error_str = str(e)
            # Check if exception message indicates "no results"
            if _is_no_results_response(error_str):
                logger.info(f"[{node_name}] Search exception indicates zero results - moving to next config")
                return {
                    "current_search_results": None,
                    "search_successful": False,
                    "retry_count": 0,
                    "search_index": search_index + 1,
                }
            # Actual error - retry
            logger.error(f"[{node_name}] Search attempt {retry_count + 1} failed: {e}")
            retry_count += 1

    # All retries exhausted (only for actual errors, not "no results")
    logger.warning(f"[{node_name}] All {MAX_RETRIES} retries exhausted for search config {search_index + 1}")
    return {
        "current_search_results": None,
        "search_successful": False,
        "retry_count": 0,
        "search_index": search_index + 1,
    }
