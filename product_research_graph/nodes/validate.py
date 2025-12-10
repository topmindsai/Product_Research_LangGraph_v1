"""Validate node for the Product Research workflow.

This module uses create_react_agent from langgraph.prebuilt for proper
tool execution with the Zyte MCP tool. Uses LangGraph's response_format
parameter for structured output with OpenAI's built-in JSON schema enforcement.

Supports domain-based tool routing:
- Amazon URLs -> get_product_data tool
- Other URLs -> scrape_product_optimized tool
"""

import asyncio
import json
import logging
from typing import Awaitable, Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
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
    get_zyte_product_data_tool,
    clear_mcp_caches,
    _is_mcp_connection_error,
)
from product_research_graph.prompts.templates import get_validation_prompt
from product_research_graph.utils.url_helpers import partition_urls_by_domain
from product_research.config.settings import LangGraphConfig
from product_research.schemas.models import (
    ValidationResponseSchema,
    ValidationImageExtractionAgentSchema__ValidatedPagesItem,
    ValidationImageExtractionAgentSchema__InvalidUrlItem,
)


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
    timeout: float = 120.0,
    tool_getter: Callable[[], Awaitable[BaseTool | None]] | None = None,
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

    Args:
        tool: The MCP tool to use for validation
        prompt: The validation prompt
        urls_str: JSON string of URLs to validate
        max_retries: Maximum number of retries for MCP connection errors
        timeout: Timeout in seconds for the agent invocation
        tool_getter: Optional function to get a fresh tool on retry. If not provided,
                     falls back to get_zyte_scrape_tool() for backward compatibility.

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

            logger.info(f"Invoking ReAct agent with Zyte tool: {current_tool.name} (attempt {attempt + 1}/{max_retries + 1}, timeout={timeout}s)")

            # Invoke the agent with timeout protection (dynamic timeout based on batch size)
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": messages}),
                timeout=timeout
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
            logger.error(f"ReAct agent validation timed out after {timeout} seconds")
            return None
        except Exception as e:
            # Check if this is an MCP connection error that should be retried
            if _is_mcp_connection_error(e) and attempt < max_retries:
                logger.warning(
                    f"MCP connection error in ReAct agent validation (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                logger.info("Clearing MCP cache and retrying with fresh connection...")
                await clear_mcp_caches()
                # Get fresh tool after clearing cache using provided getter or default
                if tool_getter:
                    fresh_tool = await tool_getter()
                else:
                    fresh_tool = await get_zyte_scrape_tool()
                if fresh_tool:
                    current_tool = fresh_tool
                    await asyncio.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    logger.error("Failed to get fresh tool after cache clear")
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


def _mark_urls_invalid(urls: list[str], reasoning: str) -> dict:
    """Mark all URLs in batch as invalid with given reasoning."""
    return {
        "validated_pages": [],
        "invalid_urls": [InvalidUrlDict(url=url, reasoning=reasoning) for url in urls],
        "total_validated_images": 0,
        "total_checked": len(urls),
    }


def _convert_result_to_dict(result: ValidationResponseSchema) -> dict:
    """Convert Pydantic ValidationResponseSchema to dict for state update."""
    validated_pages = [
        _convert_to_validated_page_dict(page)
        for page in result.validated_pages
    ]
    invalid_urls = [
        _convert_to_invalid_url_dict(item)
        for item in result.invalid_urls
    ]
    return {
        "validated_pages": validated_pages,
        "invalid_urls": invalid_urls,
        "total_validated_images": result.total_validated_images,
        "total_checked": result.total_checked,
    }


async def _process_url_group_with_tool(
    urls: list[str],
    tool: BaseTool | None,
    tool_getter: Callable[[], Awaitable[BaseTool | None]],
    tool_name: str,
    prompt: str,
    timeout: float,
) -> dict:
    """
    Process a group of URLs with a specific MCP tool.

    This helper function handles validation of a URL group using either
    the get_product_data tool (for Amazon URLs) or the scrape_product_optimized
    tool (for other URLs).

    Args:
        urls: List of URLs to validate
        tool: The MCP tool to use (or None if not available)
        tool_getter: Function to get fresh tool on retry
        tool_name: Human-readable name of the tool for logging
        prompt: The validation prompt
        timeout: Timeout in seconds

    Returns:
        Dict with validated_pages, invalid_urls, total_validated_images, total_checked
    """
    urls_str = json.dumps(urls)

    if tool is None:
        logger.error(f"{tool_name} not available, marking {len(urls)} URLs as failed")
        return _mark_urls_invalid(urls, f"{tool_name} not available")

    logger.info(f"Processing {len(urls)} URLs with {tool_name}: {tool.name}")

    try:
        structured_result = await _execute_validation_with_react_agent(
            tool=tool,
            prompt=prompt,
            urls_str=urls_str,
            timeout=timeout,
            tool_getter=tool_getter,
        )

        if structured_result is not None:
            logger.info(
                f"{tool_name} validation complete: {len(structured_result.validated_pages)} valid pages, "
                f"{structured_result.total_validated_images} images"
            )
            return _convert_result_to_dict(structured_result)

        logger.warning(f"{tool_name} validation returned no structured response")
        return _mark_urls_invalid(urls, f"{tool_name} validation failed to return structured response")

    except asyncio.TimeoutError:
        logger.error(f"{tool_name} validation timed out after {timeout} seconds")
        return _mark_urls_invalid(urls, f"{tool_name} validation timed out after {timeout}s")
    except Exception as e:
        logger.error(f"{tool_name} validation error: {e}")
        return _mark_urls_invalid(urls, f"{tool_name} validation error: {str(e)}")


async def _process_url_batch(
    urls: list[str],
    state: ProductResearchState,
) -> dict:
    """
    Process a batch of URLs with domain-based tool routing.

    Routes URLs to appropriate tools based on domain:
    - Amazon URLs -> get_product_data tool
    - Other URLs -> scrape_product_optimized tool

    Args:
        urls: List of URLs to validate in this batch
        state: Current workflow state with product info

    Returns:
        Dict with validated_pages, invalid_urls, total_validated_images, total_checked
    """
    # Calculate dynamic timeout based on batch size
    base_timeout = LangGraphConfig.VALIDATION_TIMEOUT_BASE
    per_url_timeout = LangGraphConfig.VALIDATION_TIMEOUT_PER_URL
    timeout = base_timeout + (len(urls) * per_url_timeout)

    logger.info(f"Processing batch of {len(urls)} URLs with {timeout}s timeout")

    # Get product info from state
    barcode = state.get("barcode", "")
    sku = state.get("sku", "")
    title = state.get("title", "")
    search_type_label = state.get("search_type_label", "barcode")

    # Partition URLs by domain (Amazon vs other)
    amazon_urls, other_urls = partition_urls_by_domain(urls)

    logger.info(f"URL partition: {len(amazon_urls)} Amazon URLs, {len(other_urls)} other URLs")

    # Initialize result accumulators
    all_validated_pages: list[ValidatedPageDict] = []
    all_invalid_urls: list[InvalidUrlDict] = []
    total_images = 0
    total_checked = 0

    try:
        # Process Amazon URLs with get_product_data tool
        if amazon_urls:
            amazon_tool = await get_zyte_product_data_tool()

            # Create prompt for Amazon URLs
            amazon_prompt = get_validation_prompt(
                barcode=barcode,
                sku=sku,
                title=title,
                urls=json.dumps(amazon_urls),
                search_type=search_type_label,
            )

            amazon_result = await _process_url_group_with_tool(
                urls=amazon_urls,
                tool=amazon_tool,
                tool_getter=get_zyte_product_data_tool,
                tool_name="Amazon product data tool",
                prompt=amazon_prompt,
                timeout=timeout,
            )

            all_validated_pages.extend(amazon_result.get("validated_pages", []))
            all_invalid_urls.extend(amazon_result.get("invalid_urls", []))
            total_images += amazon_result.get("total_validated_images", 0)
            total_checked += amazon_result.get("total_checked", 0)

        # Process non-Amazon URLs with scrape_product_optimized tool
        if other_urls:
            scrape_tool = await get_zyte_scrape_tool()

            # Create prompt for other URLs
            other_prompt = get_validation_prompt(
                barcode=barcode,
                sku=sku,
                title=title,
                urls=json.dumps(other_urls),
                search_type=search_type_label,
            )

            other_result = await _process_url_group_with_tool(
                urls=other_urls,
                tool=scrape_tool,
                tool_getter=get_zyte_scrape_tool,
                tool_name="Zyte scrape tool",
                prompt=other_prompt,
                timeout=timeout,
            )

            all_validated_pages.extend(other_result.get("validated_pages", []))
            all_invalid_urls.extend(other_result.get("invalid_urls", []))
            total_images += other_result.get("total_validated_images", 0)
            total_checked += other_result.get("total_checked", 0)

        logger.info(
            f"Batch validation complete: {len(all_validated_pages)} valid pages, "
            f"{total_images} images"
        )

        return {
            "validated_pages": all_validated_pages,
            "invalid_urls": all_invalid_urls,
            "total_validated_images": total_images,
            "total_checked": total_checked,
        }

    except asyncio.TimeoutError:
        logger.error(f"Batch validation timed out after {timeout} seconds")
        return _mark_urls_invalid(urls, f"Validation timed out after {timeout}s")
    except Exception as e:
        logger.error(f"Batch validation error: {e}")
        return _mark_urls_invalid(urls, f"Validation error: {str(e)}")


@traceable(name="validate_node")
async def validate_node(state: ProductResearchState) -> dict:
    """
    Validate filtered URLs and extract product images.

    This node processes URLs in batches to prevent timeout issues:
    1. Splits filtered URLs into batches (default: 3 URLs per batch)
    2. Each batch is processed with dynamic timeout (60s base + 180s per URL)
    3. Results are accumulated across batches
    4. Early exit when validated images are found (optimization)
    5. Partial results are preserved if later batches fail

    Uses LangGraph's response_format parameter for OpenAI's built-in
    JSON schema enforcement, eliminating manual JSON parsing.

    Args:
        state: Current workflow state with filtered URLs

    Returns:
        Dict with validated_pages, invalid_urls, total_validated_images, total_checked
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

    # Get batch size from config
    batch_size = LangGraphConfig.VALIDATION_BATCH_SIZE
    total_batches = (len(filtered_urls) + batch_size - 1) // batch_size

    logger.info(f"Validating {len(filtered_urls)} URLs in {total_batches} batch(es) of {batch_size}")

    # Accumulate results across batches
    all_validated_pages: list[ValidatedPageDict] = []
    all_invalid_urls: list[InvalidUrlDict] = []
    total_images = 0
    total_checked = 0

    # Process URLs in batches
    for i in range(0, len(filtered_urls), batch_size):
        batch_urls = filtered_urls[i:i + batch_size]
        batch_num = i // batch_size + 1

        logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch_urls)} URLs")

        batch_result = await _process_url_batch(
            urls=batch_urls,
            state=state,
        )

        # Accumulate results from this batch
        all_validated_pages.extend(batch_result.get("validated_pages", []))
        all_invalid_urls.extend(batch_result.get("invalid_urls", []))
        total_images += batch_result.get("total_validated_images", 0)
        total_checked += batch_result.get("total_checked", 0)

        logger.info(
            f"Batch {batch_num} complete: {len(batch_result.get('validated_pages', []))} valid pages, "
            f"{batch_result.get('total_validated_images', 0)} images. "
            f"Running total: {len(all_validated_pages)} pages, {total_images} images"
        )

        # Early exit if enabled and we found validated images (optimization)
        if LangGraphConfig.VALIDATION_EARLY_EXIT and total_images > 0:
            logger.info(f"Found {total_images} images, stopping batch processing early (early_exit=True)")
            # Mark remaining URLs as skipped (not invalid, just not processed)
            remaining_urls = filtered_urls[i + batch_size:]
            if remaining_urls:
                logger.info(f"Skipping {len(remaining_urls)} remaining URLs")
                for url in remaining_urls:
                    all_invalid_urls.append(
                        InvalidUrlDict(url=url, reasoning="Skipped - sufficient images found in previous batch")
                    )
                total_checked += len(remaining_urls)
            break

    logger.info(
        f"Validation complete: {len(all_validated_pages)} valid pages, "
        f"{total_images} images, {total_checked} URLs checked"
    )

    # Return accumulated results (state uses `add` reducer for accumulation)
    return {
        "validated_pages": all_validated_pages,
        "invalid_urls": all_invalid_urls,
        "total_validated_images": total_images,
        "total_checked": total_checked,
    }
