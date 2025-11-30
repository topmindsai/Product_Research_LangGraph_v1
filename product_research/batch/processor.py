"""Batch processing functionality for multiple products."""

import os
import asyncio
import json
from datetime import datetime
from typing import Optional

import pandas as pd

from product_research.schemas.models import ProductInput, BatchResult
from product_research_graph.workflow import run_workflow


def parse_input_file(file_path: str) -> list[ProductInput]:
    """
    Parse CSV or Excel file into list of ProductInput objects.

    Supports .csv, .xlsx, and .xls files.
    Expected columns: barcode, sku, title (case-insensitive)
    """
    file_ext = file_path.lower().split('.')[-1]

    # Read all columns as strings to preserve leading zeros in barcodes/SKUs
    if file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(file_path, dtype=str)
    elif file_ext == 'csv':
        df = pd.read_csv(file_path, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv, .xlsx, or .xls")

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Check for required columns
    required_cols = {'barcode', 'sku', 'title'}
    available_cols = set(df.columns)
    missing_cols = required_cols - available_cols

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. File must have: barcode, sku, title")

    # Convert DataFrame rows to ProductInput objects
    products = []
    for _, row in df.iterrows():
        products.append(ProductInput(
            barcode=str(row.get('barcode', '')).strip() if pd.notna(row.get('barcode')) else '',
            sku=str(row.get('sku', '')).strip() if pd.notna(row.get('sku')) else '',
            title=str(row.get('title', '')).strip() if pd.notna(row.get('title')) else ''
        ))

    return products


def write_results_to_csv(results: list[BatchResult], output_path: str) -> str:
    """
    Write batch results to CSV file.

    Returns the absolute path to the output file.
    """
    df = pd.DataFrame([r.model_dump() for r in results])
    df.to_csv(output_path, index=False)
    return os.path.abspath(output_path)


async def process_single_product_with_retry(
    product: ProductInput,
    max_retries: int = 1,
    product_index: int = 0,
    total_products: int = 0
) -> BatchResult:
    """
    Process a single product with retry logic.

    Retries once on failure, then returns error in result field.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            print(f"  Processing product {product_index + 1}/{total_products}: {product.sku or product.barcode or product.title[:30]}...")
            result = await run_workflow(product)
            result_json = result.model_dump_json()
            print(f"  Completed product {product_index + 1}/{total_products}: Found {result.total_validated_images} images")
            return BatchResult(
                barcode=product.barcode,
                sku=product.sku,
                title=product.title,
                result=result_json
            )
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                print(f"  Retry {attempt + 1}/{max_retries} for product {product_index + 1}: {last_error}")
                await asyncio.sleep(1)  # Brief delay before retry

    # All retries exhausted, return error
    error_result = {"error": last_error, "status": "failed"}
    print(f"  Failed product {product_index + 1}/{total_products}: {last_error}")
    return BatchResult(
        barcode=product.barcode,
        sku=product.sku,
        title=product.title,
        result=json.dumps(error_result)
    )


async def run_batch_workflow(
    products: list[ProductInput],
    max_concurrent: int = 3,
    output_path: Optional[str] = None
) -> tuple[list[BatchResult], str]:
    """
    Process multiple products concurrently with limited parallelism.

    Args:
        products: List of ProductInput objects to process
        max_concurrent: Maximum number of concurrent workflows (default: 3)
        output_path: Path for output CSV file (auto-generated if None)

    Returns:
        Tuple of (results list, output file path)
    """
    if not products:
        raise ValueError("No products to process")

    semaphore = asyncio.Semaphore(max_concurrent)
    total_products = len(products)

    print(f"\nStarting batch processing of {total_products} products with {max_concurrent} concurrent workers...\n")

    async def process_with_semaphore(product: ProductInput, index: int) -> BatchResult:
        async with semaphore:
            return await process_single_product_with_retry(
                product,
                product_index=index,
                total_products=total_products
            )

    # Create tasks for all products
    tasks = [
        process_with_semaphore(product, idx)
        for idx, product in enumerate(products)
    ]

    # Execute all tasks concurrently (limited by semaphore)
    results = await asyncio.gather(*tasks)

    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"batch_results_{timestamp}.csv"

    # Write results to CSV
    output_file = write_results_to_csv(results, output_path)

    print(f"\nBatch processing complete. Results saved to: {output_file}")

    return results, output_file


async def run_cli_batch(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Run batch processing from CLI.

    Args:
        input_file: Path to input CSV/Excel file
        output_file: Path for output CSV file (auto-generated if None)
    """
    print(f"Reading input file: {input_file}")

    try:
        products = parse_input_file(input_file)
        print(f"Found {len(products)} products to process")

        results, output_path = await run_batch_workflow(products, output_path=output_file)

        # Calculate statistics
        successful = sum(1 for r in results if '"error"' not in r.result)
        failed = len(results) - successful

        print(f"\n{'='*50}")
        print(f"Batch Processing Summary")
        print(f"{'='*50}")
        print(f"Total products: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output file: {output_path}")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"Error during batch processing: {e}")
        raise
