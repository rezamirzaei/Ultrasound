"""In-memory API error analytics and diagnostics."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Literal, cast

from ultrasound.api.models.domain import ApiErrorEventRecord


class ErrorAnalyticsService:
    """Collects recent API error events for operational dashboards."""

    def __init__(self, max_events: int = 500):
        self.max_events = max(50, int(max_events))
        self._events: deque[ApiErrorEventRecord] = deque(maxlen=self.max_events)
        self._lock = Lock()

    def record_error(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        detail: str,
        role: str | None = None,
    ) -> None:
        normalized_role: Literal["viewer", "analyst", "admin"] | None = None
        if role in {"viewer", "analyst", "admin"}:
            normalized_role = cast(Literal["viewer", "analyst", "admin"], role)

        event = ApiErrorEventRecord(
            occurred_at=datetime.now(tz=timezone.utc),
            request_id=request_id,
            method=method.upper(),
            path=path,
            status_code=int(status_code),
            detail=detail,
            role=normalized_role,
        )
        with self._lock:
            self._events.appendleft(event)

    def recent_errors(self, limit: int = 20) -> list[ApiErrorEventRecord]:
        n = max(1, min(int(limit), 200))
        with self._lock:
            return list(self._events)[:n]

    def summary(self, window_minutes: int = 60) -> dict[str, Any]:
        window = max(1, int(window_minutes))
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(minutes=window)

        with self._lock:
            events = list(self._events)

        recent_events = [event for event in events if event.occurred_at >= cutoff]
        by_status = Counter(str(event.status_code) for event in recent_events)
        by_path = Counter(event.path for event in recent_events)

        return {
            "generated_at": now,
            "window_minutes": window,
            "total_error_count": len(events),
            "recent_error_count": len(recent_events),
            "by_status": dict(by_status),
            "by_path": dict(by_path.most_common(10)),
            "last_error_at": events[0].occurred_at if events else None,
        }
