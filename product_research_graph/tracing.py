"""Trace saving utilities for LangGraph workflows."""

import json
import time
import os
from pathlib import Path
from typing import Optional
from langsmith import Client

# Default traces directory (project root / traces)
TRACES_DIR = Path(__file__).parent.parent / "traces"


def get_traces_dir() -> Path:
    """Get or create the traces output directory."""
    TRACES_DIR.mkdir(exist_ok=True)
    return TRACES_DIR


def fetch_and_save_traces(
    run_id: str,
    project_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    delay: float = 1.0
) -> int:
    """
    Fetch traces for a specific run from LangSmith and save to local file.

    Args:
        run_id: The run ID to fetch traces for (trace_id in LangSmith)
        project_name: LangSmith project name (defaults to LANGCHAIN_PROJECT env var)
        output_dir: Directory to save traces (defaults to ./traces/)
        delay: Seconds to wait before fetching (allows LangSmith to persist)

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
        trace_id=run_id
    )

    # Save to JSONL file
    output_file = output_dir / f"traces_{run_id}.jsonl"
    count = 0

    with open(output_file, "w") as f:
        for run in runs:
            f.write(json.dumps(run.dict(), default=str) + "\n")
            count += 1

    print(f"[Tracing] Saved {count} trace entries to {output_file}")
    return count


async def fetch_and_save_traces_async(
    run_id: str,
    project_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    delay: float = 1.0
) -> int:
    """
    Async version of fetch_and_save_traces.

    Args:
        run_id: The run ID to fetch traces for (trace_id in LangSmith)
        project_name: LangSmith project name (defaults to LANGCHAIN_PROJECT env var)
        output_dir: Directory to save traces (defaults to ./traces/)
        delay: Seconds to wait before fetching (allows LangSmith to persist)

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
        lambda: fetch_and_save_traces(run_id, project_name, output_dir, delay=0)
    )
