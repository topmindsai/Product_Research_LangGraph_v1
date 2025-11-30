"""Entry point for the Product Research Agent application."""

import argparse
import asyncio

import uvicorn


def main():
    """Main entry point for CLI and server."""
    parser = argparse.ArgumentParser(
        description="Product Image Research Agent - Find and validate product images from web sources"
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Input CSV/Excel file for batch processing"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output CSV file path (default: batch_results_TIMESTAMP.csv)"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as API server (default if no --batch provided)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )

    args = parser.parse_args()

    if args.batch:
        # Run batch processing mode
        from product_research.batch.processor import run_cli_batch
        asyncio.run(run_cli_batch(args.batch, args.output))
    else:
        # Run as API server (default)
        from product_research.api.routes import app
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
