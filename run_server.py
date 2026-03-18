from __future__ import annotations

import subprocess
import sys

import click


def _run_qdrant() -> None:
    commands = [
        ["docker", "compose", "up", "-d", "qdrant"],
        ["docker-compose", "up", "-d", "qdrant"],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
            return
    raise RuntimeError("Failed to start Qdrant. Install Docker Compose and run Docker Desktop.")


@click.command(help="Start backend services")
@click.option(
    "--ingest-mode",
    type=click.Choice(["full", "sync", "full_then_sync", "none"]),
    default="sync",
    help="Which ingestion mode to run after Qdrant starts",
)
@click.option(
    "--start-date",
    type=str,
    default=None,
    help="Only ingest messages newer than this date (format: YYYY-MM-DD)",
)
def main(ingest_mode: str, start_date: str | None) -> None:
    _run_qdrant()
    print("Qdrant is up.")

    if ingest_mode != "none":
        cmd = [sys.executable, "-m", "src.ingest", "--mode", ingest_mode]
        if start_date:
            cmd.extend(["--start-date", start_date])
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
