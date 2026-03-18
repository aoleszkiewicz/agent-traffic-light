#!/usr/bin/env python3
"""CLI script to start the FastAPI server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn


def main():
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
