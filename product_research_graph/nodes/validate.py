"""Validate node for the Product Research workflow.

This module uses create_react_agent from langgraph.prebuilt for proper
tool execution with the Zyte MCP tool.
"""

import json
import logging
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langsmith import traceable
from langgraph.prebuilt import create_react_agent

from product_research_graph.state import ProductResearchState, ValidatedPageDict
from product_research_graph.tools.mcp_tools import (
    get_zyte_scrape_tool,
    ZYTE_SCRAPE_TOOL_NAME,
)
from product_research_graph.prompts.templates import get_validation_prompt
from product_research.config.settings import LangGraphConfig


# Set up logging
logger = logging.getLogger(__name__)


def _create_validation_model():
    """
    Create the LLM model for validation based on configuration.

    Auto-detects provider from model name:
    - gpt-* → OpenAI (ChatOpenAI)
    - claude-* → Anthropic (ChatAnthropic)
    - gemini-* → Google (ChatGoogleGenerativeAI)

    Returns:
        A LangChain chat model instance configured for validation.
    """
    model_name = LangGraphConfig.VALIDATION_MODEL

    if model_name.startswith("claude-"):
        # Anthropic Claude model
        logger.info(f"Using Anthropic model: {model_name}")
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            max_tokens=4096,
        )
    elif model_name.startswith("gemini-"):
        # Google Gemini model
        logger.info(f"Using Google Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_output_tokens=4096,
        )
    else:
        # Default to OpenAI (gpt-* or any other)
        logger.info(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            use_responses_api=True,
            output_version="responses/v1",
        )


def _extract_text_from_message(message) -> str:
    """
    Extract text from AIMessage, handling both string and content block formats.

    With output_version="responses/v1", AIMessage.content may be a list of
    content blocks instead of a plain string.
    """
    # Prefer .text property if available (Responses API convenience)
    if hasattr(message, 'text') and message.text:
        return message.text

    content = message.content

    # If content is already a string, return it
    if isinstance(content, str):
        return content

    # If content is a list of content blocks, extract text
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    return str(content) if content else ""


def _extract_json_from_response(raw_results: str) -> str | None:
    """
    Extract JSON from LLM response that may contain reasoning text.

    Handles cases where JSON is:
    1. Wrapped in ```json ... ``` markdown blocks anywhere in the text
    2. Raw JSON object embedded in text
    3. Plain JSON response
    """
    content = raw_results.strip()

    # Method 1: Extract JSON from markdown code block anywhere in text
    json_match = re.search(r'```json\s*([\s\S]*?)```', content)
    if json_match:
        return json_match.group(1).strip()

    # Method 2: Try plain ``` blocks
    plain_match = re.search(r'```\s*([\s\S]*?)```', content)
    if plain_match:
        extracted = plain_match.group(1).strip()
        # Verify it looks like JSON
        if extracted.startswith('{'):
            return extracted

    # Method 3: Find raw JSON object by locating matching braces
    json_start = content.find('{')
    if json_start >= 0:
        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(content[json_start:], start=json_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return content[json_start:i + 1]

    # Method 4: If content is already valid JSON
    if content.startswith('{'):
        return content

    return None


def _parse_validation_results(raw_results: str | None) -> dict[str, Any] | None:
    """Parse validation results from string to dict."""
    if not raw_results:
        return None

    try:
        # Extract JSON from response (handles reasoning text + markdown)
        json_content = _extract_json_from_response(raw_results)

        if not json_content:
            logger.warning("Could not extract JSON from validation response")
            logger.debug(f"Raw response (first 500 chars): {raw_results[:500]}")
            return None

        parsed = json.loads(json_content)

        # Validate structure
        if isinstance(parsed, dict):
            logger.info(f"Successfully parsed validation results")
            return parsed

        logger.warning("Parsed validation result is not a dict")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in validation results: {e}")
        logger.debug(f"Raw response (first 500 chars): {raw_results[:500] if raw_results else 'None'}")
        return None


async def _execute_validation_with_react_agent(
    tool,
    prompt: str,
    urls_str: str,
) -> str | None:
    """
    Execute validation using a ReAct agent for proper tool execution.

    This uses LangGraph's create_react_agent which properly handles:
    - Tool binding to the model
    - Tool call execution
    - Multi-turn conversations
    """
    if tool is None:
        logger.error("Cannot execute ReAct agent without a tool")
        return None

    try:
        # Create the model based on configuration (supports OpenAI and Anthropic)
        model = _create_validation_model()

        # Create a ReAct agent with the tool
        agent = create_react_agent(
            model=model,
            tools=[tool],
        )

        # Prepare the input messages
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Please validate these URLs and extract product images: {urls_str}"),
        ]

        logger.info(f"Invoking ReAct agent with Zyte tool: {tool.name}")

        # Invoke the agent
        result = await agent.ainvoke({"messages": messages})

        # Extract the final response from the agent
        response_messages = result.get("messages", [])

        # Find the last AI message (the final response)
        final_response = None
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and msg.content:
                # Skip tool call responses (they have tool_calls attribute)
                if not getattr(msg, 'tool_calls', None):
                    final_response = _extract_text_from_message(msg)
                    break

        if final_response:
            logger.info(f"ReAct agent returned validation response ({len(final_response)} chars)")
            return final_response
        else:
            # If no plain AI message, get any content
            for msg in reversed(response_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    return _extract_text_from_message(msg)

        logger.warning("ReAct agent returned no usable response")
        return None

    except Exception as e:
        logger.error(f"ReAct agent validation failed: {e}")
        raise


@traceable(name="validate_node")
async def validate_node(state: ProductResearchState) -> dict:
    """
    Validate filtered URLs and extract product images.

    This node:
    1. Takes the filtered URLs from the previous node
    2. Gets the Zyte MCP tool with proper session management
    3. Uses create_react_agent for proper tool execution
    4. Validates pages by looking for barcode or SKU
    5. Extracts image URLs from validated pages

    Args:
        state: Current workflow state with filtered URLs

    Returns:
        Dict with validated_pages, invalid_urls, total_validated_images
    """
    # Check if we have URLs to validate
    filtered_urls = state.get("filtered_urls", [])
    total_filtered_urls = state.get("total_filtered_urls", 0)

    if not filtered_urls or total_filtered_urls == 0:
        logger.info("No URLs to validate")
        return {
            "validated_pages": [],
            "invalid_urls": [],
            "total_validated_images": state.get("total_validated_images", 0),
            "total_checked": state.get("total_checked", 0),
        }

    logger.info(f"Validating {len(filtered_urls)} URLs")

    # Get product info
    barcode = state.get("barcode", "")
    sku = state.get("sku", "")
    title = state.get("title", "")
    search_type_label = state.get("search_type_label", "barcode")

    # Format URLs for prompt
    urls_str = json.dumps(filtered_urls)

    # Get the validation prompt
    prompt = get_validation_prompt(
        barcode=barcode,
        sku=sku,
        title=title,
        urls=urls_str,
        search_type=search_type_label,
    )

    try:
        # Get the Zyte scrape tool with proper session management
        scrape_tool = await get_zyte_scrape_tool()

        if scrape_tool:
            logger.info(f"Got Zyte scrape tool: {scrape_tool.name}")
            # Use ReAct agent for proper tool execution
            raw_result = await _execute_validation_with_react_agent(
                tool=scrape_tool,
                prompt=prompt,
                urls_str=urls_str,
            )
        else:
            # No scrape tool available, use model only (fallback)
            logger.warning("Zyte scrape tool not available, using model-only validation")
            model = _create_validation_model()
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=f"Please validate these URLs: {urls_str}\n\n"
                    "Note: The scraping tool is not available. Please analyze the URLs "
                    "based on the URL patterns and provide your best assessment."
                ),
            ]
            response = await model.ainvoke(messages)
            raw_result = _extract_text_from_message(response)

        # Parse the results
        parsed = _parse_validation_results(raw_result)

        if parsed:
            validated_pages_raw = parsed.get("validated_pages", [])
            invalid_urls_raw = parsed.get("invalid_urls", [])
            total_validated_images = int(parsed.get("total_validated_images", 0))

            logger.info(f"Parsed validation: {len(validated_pages_raw)} valid pages, {total_validated_images} images")

            # Convert to proper format
            validated_pages: list[ValidatedPageDict] = []
            for page in validated_pages_raw:
                validated_pages.append(
                    ValidatedPageDict(
                        url=page.get("url", ""),
                        validation_method=page.get("validation_method", "unknown"),
                        image_urls=page.get("image_urls", []),
                    )
                )

            # Update total checked
            current_total_checked = state.get("total_checked", 0)
            new_total_checked = current_total_checked + len(filtered_urls)

            return {
                "validated_pages": validated_pages,
                "invalid_urls": invalid_urls_raw,
                "total_validated_images": total_validated_images,
                "total_checked": new_total_checked,
            }

        # Parsing failed
        logger.warning("Failed to parse validation results")
        return {
            "validated_pages": [],
            "invalid_urls": filtered_urls,  # Mark all as invalid if parsing fails
            "total_validated_images": state.get("total_validated_images", 0),
            "total_checked": state.get("total_checked", 0) + len(filtered_urls),
        }

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "validated_pages": [],
            "invalid_urls": filtered_urls,
            "total_validated_images": state.get("total_validated_images", 0),
            "total_checked": state.get("total_checked", 0) + len(filtered_urls),
        }
