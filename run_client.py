from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", "src/app.py"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
