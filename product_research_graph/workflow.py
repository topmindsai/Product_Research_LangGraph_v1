"""Workflow entry point for the LangGraph implementation."""

import os
from langsmith import traceable, get_current_run_tree

from product_research.schemas.models import (
    ProductInput,
    ValidationImageExtractionAgentSchema,
    ValidationImageExtractionAgentSchema__Product,
    ValidationImageExtractionAgentSchema__ValidatedPagesItem,
    ValidationImageExtractionAgentSchema__InvalidUrlItem,
    WeightSchema,
    ProductDimensionsSchema,
)
from product_research_graph.state import create_initial_state
from product_research_graph.agent import get_graph
from product_research.config.settings import LangGraphConfig
from product_research_graph.tracing import fetch_and_save_traces_async


@traceable(name="product_research_workflow")
async def run_workflow(product_input: ProductInput) -> ValidationImageExtractionAgentSchema:
    """
    Run the product research workflow using LangGraph.

    This is the main entry point for the product research system.
    Automatically saves LangSmith traces to ./traces/ after completion.

    Args:
        product_input: ProductInput with barcode, sku, and title

    Returns:
        ValidationImageExtractionAgentSchema with search results
    """
    # Capture run_id from the current trace context
    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree else None

    # Create initial state
    initial_state = create_initial_state(
        barcode=product_input.barcode or "",
        sku=product_input.sku or "",
        title=product_input.title or "",
    )

    # Get the compiled graph
    graph = get_graph()

    # Execute the workflow with increased recursion limit
    # Default is 25, but we need up to 26 steps for 8 search configs
    result = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": LangGraphConfig.RECURSION_LIMIT}
    )

    # Save traces to local file (after workflow completes)
    if run_id:
        try:
            await fetch_and_save_traces_async(run_id, delay=1.0)
        except Exception as e:
            # Don't fail the workflow if trace saving fails
            print(f"[Tracing] Warning: Failed to save traces: {e}")

    # Extract the final result
    final_result = result.get("final_result")

    if final_result:
        # Convert validated_pages to Pydantic models
        validated_pages = [
            ValidationImageExtractionAgentSchema__ValidatedPagesItem(
                url=page.get("url", ""),
                validation_method=page.get("validation_method", "unknown"),
                image_urls=page.get("image_urls", []),
                reasoning=page.get("reasoning", ""),
                product_description=page.get("product_description", ""),
                brand=page.get("brand", ""),
                weight=WeightSchema(
                    unit_of_measure=page.get("weight", {}).get("unit_of_measure", ""),
                    value=page.get("weight", {}).get("value"),
                ),
                product_dimensions=ProductDimensionsSchema(
                    length=page.get("product_dimensions", {}).get("length"),
                    width=page.get("product_dimensions", {}).get("width"),
                    height=page.get("product_dimensions", {}).get("height"),
                ),
            )
            for page in final_result.get("validated_pages", [])
        ]

        # Convert invalid_urls to Pydantic models (handle both dict and string formats)
        invalid_urls = [
            ValidationImageExtractionAgentSchema__InvalidUrlItem(
                url=item.get("url", "") if isinstance(item, dict) else item,
                reasoning=item.get("reasoning", "") if isinstance(item, dict) else "",
            )
            for item in final_result.get("invalid_urls", [])
        ]

        return ValidationImageExtractionAgentSchema(
            product=ValidationImageExtractionAgentSchema__Product(
                barcode=final_result.get("product", {}).get("barcode", ""),
                title=final_result.get("product", {}).get("title", ""),
                sku=final_result.get("product", {}).get("sku", ""),
            ),
            search_type=final_result.get("search_type", ""),
            total_checked=final_result.get("total_checked", 0),
            total_validated_images=final_result.get("total_validated_images", 0),
            validated_pages=validated_pages,
            invalid_urls=invalid_urls,
        )

    # Return empty result if no final_result
    return ValidationImageExtractionAgentSchema(
        product=ValidationImageExtractionAgentSchema__Product(
            barcode=product_input.barcode or "",
            title=product_input.title or "",
            sku=product_input.sku or "",
        ),
        search_type="barcode" if product_input.barcode else "sku",
        total_checked=0,
        total_validated_images=0,
        validated_pages=[],
        invalid_urls=[],
    )


async def run_workflow_with_streaming(product_input: ProductInput):
    """
    Run the workflow with streaming output.

    Yields state updates as the workflow progresses.

    Args:
        product_input: ProductInput with barcode, sku, and title

    Yields:
        Dict updates from each node execution
    """
    # Create initial state
    initial_state = create_initial_state(
        barcode=product_input.barcode or "",
        sku=product_input.sku or "",
        title=product_input.title or "",
    )

    # Get the compiled graph
    graph = get_graph()

    # Stream the workflow execution with increased recursion limit
    async for event in graph.astream(
        initial_state,
        config={"recursion_limit": LangGraphConfig.RECURSION_LIMIT}
    ):
        yield event
