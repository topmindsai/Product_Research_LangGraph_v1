#!/usr/bin/env python3
"""CLI script to fetch LangSmith traces for a given run ID."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from product_research_graph.tracing import fetch_and_save_traces


def main():
    parser = argparse.ArgumentParser(
        description="Fetch LangSmith traces for a given run ID"
    )
    parser.add_argument(
        "run_id",
        help="The LangSmith run/trace ID to fetch traces for"
    )
    parser.add_argument(
        "--project", "-p",
        help="LangSmith project name (defaults to LANGCHAIN_PROJECT env var)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory (defaults to ./traces/)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        help="Seconds to wait before fetching (default: 1.0)"
    )

    args = parser.parse_args()

    try:
        count = fetch_and_save_traces(
            run_id=args.run_id,
            project_name=args.project,
            output_dir=args.output,
            delay=args.delay
        )

        if count > 0:
            print(f"Successfully fetched {count} trace entries")
        else:
            print("No traces found for the given run ID")
            sys.exit(1)

    except Exception as e:
        print(f"Error fetching traces: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
