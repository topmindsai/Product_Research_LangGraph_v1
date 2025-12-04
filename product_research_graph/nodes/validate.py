"""Validate node for the Product Research workflow.

This module uses create_react_agent from langgraph.prebuilt for proper
tool execution with the Zyte MCP tool. Uses LangGraph's response_format
parameter for structured output with OpenAI's built-in JSON schema enforcement.
"""

import asyncio
import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from langgraph.prebuilt import create_react_agent

from product_research_graph.state import (
    ProductResearchState,
    ValidatedPageDict,
    InvalidUrlDict,
    WeightDict,
    ProductDimensionsDict,
)
from product_research_graph.tools.mcp_tools import (
    get_zyte_scrape_tool,
    clear_mcp_caches,
    _is_mcp_connection_error,
)
from product_research_graph.prompts.templates import get_validation_prompt
from product_research.config.settings import LangGraphConfig
from product_research.schemas.models import ValidationResponseSchema


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


async def _execute_validation_with_react_agent(
    tool,
    prompt: str,
    urls_str: str,
    max_retries: int = 2,
) -> ValidationResponseSchema | None:
    """
    Execute validation using a ReAct agent with structured output.

    This uses LangGraph's create_react_agent with response_format parameter
    which properly handles:
    - Tool binding to the model
    - Tool call execution
    - Multi-turn conversations
    - Structured output via OpenAI's built-in JSON schema enforcement

    Includes retry logic for MCP connection errors (ClosedResourceError, etc.)
    which can occur when SSE streams are unexpectedly closed.

    Returns:
        ValidationResponseSchema on success, None on failure.
    """
    if tool is None:
        logger.error("Cannot execute ReAct agent without a tool")
        return None

    current_tool = tool

    for attempt in range(max_retries + 1):
        try:
            # Create the model based on configuration (supports OpenAI, Anthropic, Gemini)
            model = _create_validation_model()

            # Create a ReAct agent with structured output
            # response_format enables OpenAI's built-in JSON schema enforcement
            agent = create_react_agent(
                model=model,
                tools=[current_tool],
                response_format=ValidationResponseSchema,
            )

            # Prepare the input messages
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Please validate these URLs and extract product images: {urls_str}"),
            ]

            logger.info(f"Invoking ReAct agent with Zyte tool: {current_tool.name} (attempt {attempt + 1}/{max_retries + 1})")

            # Invoke the agent with timeout protection (longer for validation/scraping)
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": messages}),
                timeout=120.0  # 2 minute timeout for validation
            )

            # Access the structured response directly (guaranteed to match schema)
            structured_response = result.get("structured_response")

            if structured_response is not None:
                logger.info(
                    f"ReAct agent returned structured response with "
                    f"{len(structured_response.validated_pages)} valid pages, "
                    f"{structured_response.total_validated_images} images"
                )
                return structured_response

            logger.warning("ReAct agent returned no structured response")
            return None

        except asyncio.TimeoutError:
            logger.error("ReAct agent validation timed out after 120 seconds")
            return None
        except Exception as e:
            # Check if this is an MCP connection error that should be retried
            if _is_mcp_connection_error(e) and attempt < max_retries:
                logger.warning(
                    f"MCP connection error in ReAct agent validation (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                logger.info("Clearing MCP cache and retrying with fresh connection...")
                await clear_mcp_caches()
                # Get fresh tool after clearing cache
                fresh_tool = await get_zyte_scrape_tool()
                if fresh_tool:
                    current_tool = fresh_tool
                    await asyncio.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    logger.error("Failed to get fresh Zyte tool after cache clear")
                    return None
            else:
                # Log but DON'T re-raise - return None for graceful degradation
                logger.error(f"ReAct agent validation failed: {e}")
                return None

    return None


def _convert_to_validated_page_dict(
    page: "ValidationImageExtractionAgentSchema__ValidatedPagesItem",
) -> ValidatedPageDict:
    """Convert Pydantic ValidatedPagesItem to TypedDict for state compatibility."""
    return ValidatedPageDict(
        url=page.url,
        validation_method=page.validation_method,
        image_urls=page.image_urls,
        reasoning=page.reasoning,
        product_description=page.product_description,
        brand=page.brand,
        weight=WeightDict(
            unit_of_measure=page.weight.unit_of_measure,
            value=page.weight.value,
        ),
        product_dimensions=ProductDimensionsDict(
            length=page.product_dimensions.length,
            width=page.product_dimensions.width,
            height=page.product_dimensions.height,
        ),
    )


def _convert_to_invalid_url_dict(
    item: "ValidationImageExtractionAgentSchema__InvalidUrlItem",
) -> InvalidUrlDict:
    """Convert Pydantic InvalidUrlItem to TypedDict for state compatibility."""
    return InvalidUrlDict(
        url=item.url,
        reasoning=item.reasoning,
    )


@traceable(name="validate_node")
async def validate_node(state: ProductResearchState) -> dict:
    """
    Validate filtered URLs and extract product images.

    This node:
    1. Takes the filtered URLs from the previous node
    2. Gets the Zyte MCP tool with proper session management
    3. Uses create_react_agent with response_format for structured output
    4. Validates pages by looking for barcode or SKU
    5. Extracts image URLs from validated pages

    Uses LangGraph's response_format parameter for OpenAI's built-in
    JSON schema enforcement, eliminating manual JSON parsing.

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
        # Return zeros - with `add` reducer, this adds nothing to current totals
        return {
            "validated_pages": [],
            "invalid_urls": [],
            "total_validated_images": 0,
            "total_checked": 0,
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
            # Use ReAct agent with structured output
            structured_result = await _execute_validation_with_react_agent(
                tool=scrape_tool,
                prompt=prompt,
                urls_str=urls_str,
            )
        else:
            # No scrape tool available, use model with structured output directly
            logger.warning("Zyte scrape tool not available, using model-only validation")
            model = _create_validation_model()

            # Use with_structured_output for consistency
            structured_model = model.with_structured_output(ValidationResponseSchema)

            messages = [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=f"Please validate these URLs: {urls_str}\n\n"
                    "Note: The scraping tool is not available. Please analyze the URLs "
                    "based on the URL patterns and provide your best assessment."
                ),
            ]
            structured_result = await structured_model.ainvoke(messages)

        # Process structured result
        if structured_result is not None:
            logger.info(
                f"Validation complete: {len(structured_result.validated_pages)} valid pages, "
                f"{structured_result.total_validated_images} images"
            )

            # Convert Pydantic models to TypedDicts for state compatibility
            validated_pages = [
                _convert_to_validated_page_dict(page)
                for page in structured_result.validated_pages
            ]

            invalid_urls = [
                _convert_to_invalid_url_dict(item)
                for item in structured_result.invalid_urls
            ]

            # Return incremental counts (state uses `add` reducer for accumulation)
            return {
                "validated_pages": validated_pages,
                "invalid_urls": invalid_urls,
                "total_validated_images": structured_result.total_validated_images,
                "total_checked": structured_result.total_checked,
            }

        # Structured output failed
        logger.warning("Failed to get structured validation results")
        return {
            "validated_pages": [],
            "invalid_urls": [
                InvalidUrlDict(url=url, reasoning="Validation failed to return structured response")
                for url in filtered_urls
            ],
            "total_validated_images": 0,
            "total_checked": len(filtered_urls),
        }

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "validated_pages": [],
            "invalid_urls": [
                InvalidUrlDict(url=url, reasoning=f"Validation error: {str(e)}")
                for url in filtered_urls
            ],
            "total_validated_images": 0,
            "total_checked": len(filtered_urls),
        }
