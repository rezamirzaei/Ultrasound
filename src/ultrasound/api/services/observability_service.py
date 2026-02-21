"""Application metrics and lightweight observability helpers."""

from __future__ import annotations

from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class ObservabilityService:
    """Collects request and background-job metrics for production monitoring."""

    METRICS_CONTENT_TYPE: Final[str] = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)

        self.http_requests_total = Counter(
            "inphase_http_requests_total",
            "Total HTTP requests handled by API",
            labelnames=("method", "route", "status"),
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            "inphase_http_request_duration_seconds",
            "HTTP request latency in seconds",
            labelnames=("method", "route", "status"),
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self.http_exceptions_total = Counter(
            "inphase_http_exceptions_total",
            "Unhandled exceptions before response mapping",
            labelnames=("route", "exception_type"),
            registry=self.registry,
        )

        self.job_runs_total = Counter(
            "inphase_job_runs_total",
            "Background jobs by type and final status",
            labelnames=("job_type", "status"),
            registry=self.registry,
        )
        self.job_duration_seconds = Histogram(
            "inphase_job_duration_seconds",
            "Background job duration in seconds",
            labelnames=("job_type",),
            registry=self.registry,
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 3.0, 10.0, 30.0, 60.0, 120.0),
        )
        self.worker_up = Gauge(
            "inphase_job_worker_up",
            "Job worker process state (1=running, 0=stopped)",
            registry=self.registry,
        )

    def observe_http_request(
        self, method: str, route: str, status_code: int, duration_seconds: float
    ) -> None:
        labels = {
            "method": method.upper(),
            "route": route,
            "status": str(int(status_code)),
        }
        self.http_requests_total.labels(**labels).inc()
        self.http_request_duration_seconds.labels(**labels).observe(
            max(0.0, float(duration_seconds))
        )

    def observe_http_exception(self, route: str, exception_type: str) -> None:
        self.http_exceptions_total.labels(route=route, exception_type=exception_type).inc()

    def observe_job(self, job_type: str, status: str, duration_seconds: float) -> None:
        self.job_runs_total.labels(job_type=job_type, status=status).inc()
        self.job_duration_seconds.labels(job_type=job_type).observe(
            max(0.0, float(duration_seconds))
        )

    def set_worker_up(self, is_up: bool) -> None:
        self.worker_up.set(1.0 if is_up else 0.0)

    def render_metrics(self) -> bytes:
        return generate_latest(self.registry)
