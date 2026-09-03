"""Run the single-worker inference service."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "claim_detector.api.app:app",
        host=os.environ.get("CLAIM_HOST", "127.0.0.1"),
        port=int(os.environ.get("CLAIM_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
