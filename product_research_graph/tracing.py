"""Trace saving utilities for LangGraph workflows."""

import csv
import json
import time
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from langsmith import Client

# Supported output formats
OutputFormat = Literal["jsonl", "json", "csv"]

# Default traces directory (project root / traces)
TRACES_DIR = Path(__file__).parent.parent / "traces"


def get_traces_dir() -> Path:
    """Get or create the traces output directory."""
    TRACES_DIR.mkdir(exist_ok=True)
    return TRACES_DIR


def fetch_and_save_traces(
    trace_id: str,
    project_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    delay: float = 1.0,
    output_format: OutputFormat = "jsonl"
) -> int:
    """
    Fetch traces for a specific trace from LangSmith and save to local file.

    Args:
        trace_id: The LangSmith trace ID to fetch
        project_name: LangSmith project name (defaults to LANGCHAIN_PROJECT env var)
        output_dir: Directory to save traces (defaults to ./traces/)
        delay: Seconds to wait before fetching (allows LangSmith to persist)
        output_format: Output format - "jsonl", "json", or "csv" (default: "jsonl")

    Returns:
        Number of trace entries saved
    """
    # Allow LangSmith time to persist traces
    if delay > 0:
        time.sleep(delay)

    # Get project name from env if not provided
    if project_name is None:
        project_name = os.environ.get("LANGCHAIN_PROJECT", "default")

    # Get output directory
    if output_dir is None:
        output_dir = get_traces_dir()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Initialize LangSmith client
    client = Client()

    # Fetch all runs for this trace
    runs = client.list_runs(
        project_name=project_name,
        trace_id=trace_id
    )

    # Collect runs as dicts with warning suppression
    run_dicts = []
    for run in runs:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*Pydantic serializer warnings.*"
            )
            run_dicts.append(run.dict())

    count = len(run_dicts)
    if count == 0:
        return 0

    # Determine file extension based on format
    ext = {"jsonl": ".jsonl", "json": ".json", "csv": ".csv"}[output_format]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"traces_{timestamp}_{trace_id}{ext}"

    if output_format == "jsonl":
        with open(output_file, "w") as f:
            for run_dict in run_dicts:
                f.write(json.dumps(run_dict, default=str) + "\n")

    elif output_format == "json":
        with open(output_file, "w") as f:
            json.dump(run_dicts, f, default=str, indent=2)

    elif output_format == "csv":
        # Flatten nested fields as JSON strings for CSV
        flat_rows = []
        for run_dict in run_dicts:
            flat_row = {}
            for key, value in run_dict.items():
                if isinstance(value, (dict, list)):
                    flat_row[key] = json.dumps(value, default=str)
                else:
                    flat_row[key] = value
            flat_rows.append(flat_row)

        # Get all unique keys across all rows
        all_keys = set()
        for row in flat_rows:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    print(f"[Tracing] Saved {count} trace entries to {output_file}")
    return count


async def fetch_and_save_traces_async(
    trace_id: str,
    project_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    delay: float = 1.0,
    output_format: OutputFormat = "jsonl"
) -> int:
    """
    Async version of fetch_and_save_traces.

    Args:
        trace_id: The LangSmith trace ID to fetch
        project_name: LangSmith project name (defaults to LANGCHAIN_PROJECT env var)
        output_dir: Directory to save traces (defaults to ./traces/)
        delay: Seconds to wait before fetching (allows LangSmith to persist)
        output_format: Output format - "jsonl", "json", or "csv" (default: "jsonl")

    Returns:
        Number of trace entries saved
    """
    import asyncio

    if delay > 0:
        await asyncio.sleep(delay)

    # The actual LangSmith client calls are sync, so we run in executor
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: fetch_and_save_traces(trace_id, project_name, output_dir, delay=0, output_format=output_format)
    )
