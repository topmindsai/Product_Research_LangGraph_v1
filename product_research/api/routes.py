"""FastAPI application and routes."""

import os
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse

from product_research.schemas.models import (
    ProductInput,
    ValidationImageExtractionAgentSchema,
    BatchJobResponse,
)
from product_research_graph.workflow import run_workflow
from product_research.batch.processor import (
    parse_input_file,
    run_batch_workflow,
)


# FastAPI Application
app = FastAPI(
    title="Product Image Research API",
    description="Find and validate product images from web sources using barcode, SKU, and/or title",
    version="1.0.0"
)


@app.post("/api/v1/product-images", response_model=ValidationImageExtractionAgentSchema)
async def find_product_images(product: ProductInput):
    """
    Search for product images using barcode, SKU, and/or title.

    Returns validated product images from various web sources.
    The workflow searches multiple sources (Google, Yahoo, OpenAI web search)
    and validates that found images match the exact product variant.
    """
    try:
        result = await run_workflow(product)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/v1/product-images/batch", response_model=BatchJobResponse)
async def batch_find_product_images(
    file: UploadFile = File(..., description="CSV or Excel file with barcode, sku, title columns"),
    output_filename: Optional[str] = Query(default=None, description="Custom output filename (optional)")
):
    """
    Process multiple products from CSV or Excel file.

    Upload a file (.csv, .xlsx, .xls) with columns: barcode, sku, title
    The system will process up to 3 products concurrently.

    Returns processing results and saves output to a CSV file.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = file.filename.lower().split('.')[-1]
    if file_ext not in ['csv', 'xlsx', 'xls']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Use .csv, .xlsx, or .xls"
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Parse the input file
            products = parse_input_file(tmp_path)

            if not products:
                raise HTTPException(status_code=400, detail="No products found in file")

            # Generate output filename
            if output_filename:
                output_path = output_filename if output_filename.endswith('.csv') else f"{output_filename}.csv"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"batch_results_{timestamp}.csv"

            # Run batch workflow
            results, output_file = await run_batch_workflow(products, output_path=output_path)

            # Calculate statistics
            successful = sum(1 for r in results if '"error"' not in r.result)
            failed = len(results) - successful

            return BatchJobResponse(
                total_products=len(results),
                successful=successful,
                failed=failed,
                output_file=output_file,
                results=results
            )

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/batch-results/{filename}")
async def download_batch_results(filename: str):
    """
    Download a batch results CSV file.

    Provide the filename returned from the batch processing endpoint.
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail=f"Results file not found: {filename}")

    return FileResponse(
        path=filename,
        media_type="text/csv",
        filename=os.path.basename(filename)
    )
