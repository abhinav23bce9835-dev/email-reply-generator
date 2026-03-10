"""Benchmark runner for measuring pipeline performance."""
import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def measure_latency(
    pipeline: Any,
    test_inputs: list[Any],
) -> dict[str, float]:
    """
    Measure latency statistics for the pipeline.

    Args:
        pipeline: RAGEmailPipeline instance.
        test_inputs: List of EmailRequest objects.

    Returns:
        Dictionary with avg, p50, p95, p99 latency in milliseconds.
    """
    latencies = []

    for email_request in test_inputs:
        start = time.perf_counter()
        try:
            asyncio.get_event_loop().run_until_complete(
                pipeline.generate_reply(email_request)
            )
        except Exception as exc:
            logger.warning("Pipeline failed during benchmark: %s", exc)
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    latencies.sort()
    n = len(latencies)

    def percentile(data: list, p: float) -> float:
        """Compute percentile using linear interpolation."""
        idx = (len(data) - 1) * p
        lo = int(idx)
        hi = lo + 1
        if hi >= len(data):
            return data[lo]
        return data[lo] + (data[hi] - data[lo]) * (idx - lo)

    return {
        "avg": round(sum(latencies) / n, 2),
        "p50": round(percentile(latencies, 0.50), 2),
        "p95": round(percentile(latencies, 0.95), 2),
        "p99": round(percentile(latencies, 0.99), 2),
        "total_requests": n,
    }


def measure_throughput(
    pipeline: Any,
    test_inputs: list[Any],
    duration_seconds: float = 10.0,
) -> dict[str, float]:
    """
    Measure the throughput of the pipeline over a fixed duration.

    Args:
        pipeline: RAGEmailPipeline instance.
        test_inputs: List of EmailRequest objects (cycled as needed).
        duration_seconds: Duration to measure throughput over.

    Returns:
        Dictionary with requests_per_second and total_requests.
    """
    start_time = time.time()
    request_count = 0
    idx = 0

    while time.time() - start_time < duration_seconds:
        email_request = test_inputs[idx % len(test_inputs)]
        try:
            asyncio.get_event_loop().run_until_complete(
                pipeline.generate_reply(email_request)
            )
            request_count += 1
        except Exception as exc:
            logger.warning("Pipeline failed: %s", exc)
        idx += 1

    elapsed = time.time() - start_time
    rps = request_count / elapsed if elapsed > 0 else 0.0

    return {
        "requests_per_second": round(rps, 2),
        "total_requests": request_count,
        "elapsed_seconds": round(elapsed, 2),
    }
