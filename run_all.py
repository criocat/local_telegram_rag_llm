from __future__ import annotations

import subprocess
import sys
import time


def main() -> None:
    server_proc = subprocess.Popen([sys.executable, "run_server.py", "--ingest-mode", "sync"])
    try:
        time.sleep(4)
        subprocess.run([sys.executable, "run_client.py"], check=True)
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()


if __name__ == "__main__":
    main()
