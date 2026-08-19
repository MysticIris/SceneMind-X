"""Spawn the local SceneMind-X API with direct file-backed output streams."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Execute the main operation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    args = parser.parse_args()

    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    with args.stdout.open("ab", buffering=0) as stdout_file, args.stderr.open(
        "ab", buffering=0
    ) as stderr_file:
        process = subprocess.Popen(
            [
                str(args.python),
                "-m",
                "uvicorn",
                "apps.api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            cwd=args.project_root,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            close_fds=True,
            creationflags=creation_flags,
        )
    print(
        json.dumps(
            {
                "started": True,
                "launcher_pid": process.pid,
                "stdout_log": str(args.stdout),
                "stderr_log": str(args.stderr),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
