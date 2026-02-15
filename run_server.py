from __future__ import annotations

import argparse
import subprocess
import sys


def _run_compose_up() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Start backend services")
    parser.add_argument(
        "--ingest-mode",
        choices=["full", "sync", "full_then_sync", "none"],
        default="sync",
        help="Which ingestion mode to run after Qdrant starts",
    )
    args = parser.parse_args()

    _run_compose_up()
    print("Qdrant is up.")

    if args.ingest_mode != "none":
        cmd = [sys.executable, "-m", "src.ingest", "--mode", args.ingest_mode]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
