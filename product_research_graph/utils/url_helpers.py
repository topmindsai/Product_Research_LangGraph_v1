"""URL helper utilities for domain-based routing.

This module provides utilities for detecting and routing URLs based on domain,
primarily for differentiating Amazon URLs from other e-commerce URLs.
"""

import logging
import re
from urllib.parse import urlparse


# Set up logging
logger = logging.getLogger(__name__)


# Complete list of Amazon TLDs (24 domains)
AMAZON_DOMAINS = frozenset([
    "amazon.com",      # USA
    "amazon.ca",       # Canada
    "amazon.com.mx",   # Mexico
    "amazon.com.br",   # Brazil
    "amazon.co.uk",    # United Kingdom
    "amazon.de",       # Germany
    "amazon.fr",       # France
    "amazon.it",       # Italy
    "amazon.es",       # Spain
    "amazon.nl",       # Netherlands
    "amazon.pl",       # Poland
    "amazon.se",       # Sweden
    "amazon.com.be",   # Belgium
    "amazon.ie",       # Ireland
    "amazon.in",       # India
    "amazon.co.jp",    # Japan
    "amazon.cn",       # China
    "amazon.sg",       # Singapore
    "amazon.sa",       # Saudi Arabia
    "amazon.ae",       # United Arab Emirates
    "amazon.com.tr",   # Turkey
    "amazon.eg",       # Egypt
    "amazon.co.za",    # South Africa
    "amazon.com.au",   # Australia
])

# Pre-compile regex pattern for Amazon domain matching
# Matches: www.amazon.com, amazon.com, m.amazon.com, smile.amazon.com, etc.
# Pattern explanation:
#   ^(?:[\w-]+\.)*  - Optional subdomains (www., m., smile., etc.)
#   (amazon\.[domain])  - Core Amazon domain
#   $  - End of string
_amazon_domains_escaped = '|'.join(re.escape(d) for d in AMAZON_DOMAINS)
AMAZON_DOMAIN_PATTERN = re.compile(
    rf'^(?:[\w-]+\.)*({_amazon_domains_escaped})$',
    re.IGNORECASE
)


def is_amazon_url(url: str) -> bool:
    """
    Check if a URL belongs to an Amazon domain.

    Handles:
    - All 24 Amazon TLDs (amazon.com, amazon.co.uk, amazon.de, etc.)
    - Subdomains (www.amazon.com, m.amazon.com, smile.amazon.com)
    - URL variations with/without protocol
    - Case-insensitive matching

    Args:
        url: The URL to check

    Returns:
        True if the URL is an Amazon domain, False otherwise

    Examples:
        >>> is_amazon_url("https://www.amazon.com/dp/B08N5WRWNW")
        True
        >>> is_amazon_url("https://amazon.co.uk/product/123")
        True
        >>> is_amazon_url("https://m.amazon.de/gp/product/ABC")
        True
        >>> is_amazon_url("https://www.walmart.com/ip/product")
        False
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url)
        # Get the netloc (host:port), fall back to path for URLs without scheme
        host = parsed.netloc or parsed.path.split('/')[0]
        # Remove port if present
        host = host.split(':')[0].lower().strip()

        if not host:
            return False

        # Check if host matches Amazon pattern
        return bool(AMAZON_DOMAIN_PATTERN.match(host))
    except Exception as e:
        logger.debug(f"Error parsing URL '{url}': {e}")
        return False


def partition_urls_by_domain(urls: list[str]) -> tuple[list[str], list[str]]:
    """
    Partition URLs into Amazon and non-Amazon groups.

    This function separates a list of URLs based on whether they belong
    to Amazon domains. Amazon URLs will be routed to the get_product_data
    tool, while other URLs will use scrape_product_optimized.

    Args:
        urls: List of URLs to partition

    Returns:
        Tuple of (amazon_urls, other_urls)

    Examples:
        >>> amazon, other = partition_urls_by_domain([
        ...     "https://amazon.com/product1",
        ...     "https://walmart.com/product2",
        ...     "https://www.amazon.co.uk/product3"
        ... ])
        >>> len(amazon)
        2
        >>> len(other)
        1
    """
    amazon_urls: list[str] = []
    other_urls: list[str] = []

    for url in urls:
        if is_amazon_url(url):
            amazon_urls.append(url)
        else:
            other_urls.append(url)

    if amazon_urls or other_urls:
        logger.debug(
            f"URL partition result: {len(amazon_urls)} Amazon URLs, "
            f"{len(other_urls)} other URLs"
        )

    return amazon_urls, other_urls
