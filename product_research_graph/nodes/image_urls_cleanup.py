"""Image URLs cleanup node for the Product Research workflow.

This node cleans up image URLs in validated pages by:
1. Removing duplicate URLs within each page
2. Verifying each URL returns a valid image via HTTP GET request
3. Updating total_validated_images to reflect actual valid images
"""

import asyncio
import logging

import httpx
from langsmith import traceable

from product_research_graph.state import (
    ProductResearchState,
    ValidatedPageDict,
    WeightDict,
    ProductDimensionsDict,
)


logger = logging.getLogger(__name__)

# Image magic bytes for common formats
IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",  # WebP starts with RIFF
}

# Valid image content types
VALID_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "image/bmp",
    "image/tiff",
    "image/x-icon",
}

# Configuration
REQUEST_TIMEOUT = 10.0  # seconds per request
MAX_CONCURRENT_REQUESTS = 10
MAX_BYTES_TO_READ = 16  # bytes for magic number detection


async def _is_valid_image_url(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """
    Verify if a URL points to a valid image.

    Checks:
    1. URL returns 200 status
    2. Content-Type header indicates image
    3. First bytes match image magic numbers

    Args:
        client: httpx async client
        url: URL to verify
        semaphore: Concurrency limiter

    Returns:
        True if URL is a valid image, False otherwise
    """
    async with semaphore:
        try:
            # Use streaming to avoid downloading entire image
            async with client.stream(
                "GET",
                url,
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
            ) as response:
                # Check status code
                if response.status_code != 200:
                    return False

                # Check Content-Type header
                content_type = response.headers.get("content-type", "").lower()
                content_type_base = content_type.split(";")[0].strip()

                # Read first bytes to verify magic numbers
                first_bytes = b""
                async for chunk in response.aiter_bytes():
                    first_bytes += chunk
                    if len(first_bytes) >= MAX_BYTES_TO_READ:
                        break

                if not first_bytes:
                    return False

                # Check magic bytes
                for magic, _ in IMAGE_MAGIC_BYTES.items():
                    if first_bytes.startswith(magic):
                        return True

                # If content-type was valid but magic didn't match,
                # trust content-type (some formats have variable headers)
                if content_type_base in VALID_IMAGE_CONTENT_TYPES:
                    return True

                return False

        except (httpx.TimeoutException, httpx.RequestError, Exception):
            return False


async def _cleanup_page_images(
    client: httpx.AsyncClient,
    page: ValidatedPageDict,
    semaphore: asyncio.Semaphore,
) -> tuple[ValidatedPageDict, int]:
    """
    Clean up image URLs for a single validated page.

    Args:
        client: httpx async client
        page: Validated page dict
        semaphore: Concurrency limiter

    Returns:
        Tuple of (cleaned page, count of valid images)
    """
    original_urls = page.get("image_urls", [])

    # Step 1: Deduplicate URLs (preserve order)
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in original_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    # Step 2: Verify each URL is a valid image
    validation_tasks = [
        _is_valid_image_url(client, url, semaphore) for url in unique_urls
    ]
    results = await asyncio.gather(*validation_tasks)

    # Step 3: Keep only valid URLs
    valid_urls = [url for url, is_valid in zip(unique_urls, results) if is_valid]

    # Create cleaned page (copy all fields, update image_urls)
    cleaned_page = ValidatedPageDict(
        url=page.get("url", ""),
        validation_method=page.get("validation_method", "unknown"),
        image_urls=valid_urls,
        reasoning=page.get("reasoning", ""),
        product_description=page.get("product_description", ""),
        brand=page.get("brand", ""),
        weight=page.get("weight", WeightDict(unit_of_measure="", value=None)),
        product_dimensions=page.get(
            "product_dimensions",
            ProductDimensionsDict(length=None, width=None, height=None),
        ),
        is_shopify=page.get("is_shopify"),
    )

    return cleaned_page, len(valid_urls)


@traceable(name="image_urls_cleanup_node")
async def image_urls_cleanup_node(state: ProductResearchState) -> dict:
    """
    Clean up image URLs in validated pages.

    This node:
    1. Deduplicates image URLs within each page
    2. Verifies each URL returns a valid image via HTTP
    3. Updates total_validated_images to reflect actual valid count

    Args:
        state: Current workflow state with validated_pages

    Returns:
        Dict with cleaned_validated_pages and corrected image count
    """
    validated_pages = state.get("validated_pages", [])

    if not validated_pages:
        logger.info("No validated pages to clean up")
        return {
            "cleaned_validated_pages": [],
            "cleaned_total_validated_images": 0,
        }

    logger.info(f"Cleaning up image URLs for {len(validated_pages)} validated pages")

    # Create semaphore for concurrency limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Process all pages
    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (compatible; ProductResearchBot/1.0)"},
        follow_redirects=True,
    ) as client:
        tasks = [
            _cleanup_page_images(client, page, semaphore) for page in validated_pages
        ]
        results = await asyncio.gather(*tasks)

    # Collect results
    cleaned_pages: list[ValidatedPageDict] = []
    total_valid_images = 0

    for cleaned_page, image_count in results:
        cleaned_pages.append(cleaned_page)
        total_valid_images += image_count

    logger.info(
        f"Image cleanup complete: {total_valid_images} valid images "
        f"across {len(cleaned_pages)} pages"
    )

    return {
        "cleaned_validated_pages": cleaned_pages,
        "cleaned_total_validated_images": total_valid_images,
    }
