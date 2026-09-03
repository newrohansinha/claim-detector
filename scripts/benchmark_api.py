"""Measure end-to-end latency and throughput against a running real API."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

SENTENCES = (
    "The Empire State Building is in New York City.",
    "Please close the door.",
    "Global sea level rose during the twentieth century.",
    "What time does the train leave?",
    "A triangle has three sides.",
    "This movie is wonderful.",
    "The committee published its report on Tuesday.",
    "Imagine a city floating above the clouds.",
)


async def benchmark(
    url: str,
    *,
    requests: int,
    concurrency: int,
    warmup: int,
    timeout: float,
    deployment: str,
) -> dict[str, Any]:
    if requests < 1 or concurrency < 1 or warmup < 0:
        raise ValueError("requests and concurrency must be positive; warmup cannot be negative")
    endpoint = f"{url.rstrip('/')}/v1/predict"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        readiness = await client.get(f"{url.rstrip('/')}/health/ready")
        readiness.raise_for_status()
        model_response = await client.get(f"{url.rstrip('/')}/v1/model")
        model_response.raise_for_status()
        model_info = model_response.json()
        for index in range(warmup):
            response = await client.post(
                endpoint, json={"sentence": SENTENCES[index % len(SENTENCES)]}
            )
            response.raise_for_status()

        semaphore = asyncio.Semaphore(concurrency)

        async def one_request(index: int) -> tuple[float, dict[str, Any]]:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    endpoint,
                    json={"sentence": SENTENCES[index % len(SENTENCES)]},
                )
                latency_ms = (time.perf_counter() - started) * 1_000
                response.raise_for_status()
                body = response.json()
                expected = {"is_claim", "confidence", "claim_probability", "model_version"}
                if not expected.issubset(body):
                    raise ValueError(f"Response is missing fields: {sorted(expected - set(body))}")
                return latency_ms, body

        started = time.perf_counter()
        completed = await asyncio.gather(*(one_request(index) for index in range(requests)))
        wall_seconds = time.perf_counter() - started

    latencies = np.asarray([latency for latency, _ in completed])
    versions = sorted({str(body["model_version"]) for _, body in completed})
    return {
        "schema_version": 1,
        "measured_at": datetime.now(UTC).isoformat(),
        "target": endpoint,
        "deployment": deployment,
        "server_model": model_info,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "configuration": {
            "requests": requests,
            "concurrency": concurrency,
            "warmup_requests": warmup,
            "timeout_seconds": timeout,
            "sentence_variants": len(SENTENCES),
        },
        "results": {
            "successful_requests": len(completed),
            "wall_seconds": wall_seconds,
            "throughput_requests_per_second": len(completed) / wall_seconds,
            "latency_ms": {
                "minimum": float(latencies.min()),
                "mean": statistics.fmean(latencies),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "maximum": float(latencies.max()),
            },
            "model_versions": versions,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--deployment", default="unspecified")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = asyncio.run(
        benchmark(
            args.url,
            requests=args.requests,
            concurrency=args.concurrency,
            warmup=args.warmup,
            timeout=args.timeout,
            deployment=args.deployment,
        )
    )
    rendered = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
