"""Persistent API error analytics and diagnostics using SQLAlchemy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

from sqlalchemy import delete, func, select

from ultrasound.api.database.models import ApiErrorEventORM
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import ApiErrorEventRecord


class ErrorAnalyticsService:
    """Collects API error events for operational dashboards."""

    def __init__(self, db: DatabaseSessionManager, max_events: int = 5000):
        self.db = db
        self.max_events = max(200, int(max_events))
        self.db.create_schema()

    def _to_record(self, row: ApiErrorEventORM) -> ApiErrorEventRecord:
        normalized_role: Literal["viewer", "analyst", "admin"] | None = None
        if row.role in {"viewer", "analyst", "admin"}:
            normalized_role = cast(Literal["viewer", "analyst", "admin"], row.role)

        return ApiErrorEventRecord(
            occurred_at=row.occurred_at,
            request_id=row.request_id,
            method=row.method,
            path=row.path,
            status_code=int(row.status_code),
            detail=row.detail,
            role=normalized_role,
        )

    def _trim_old_events(self) -> None:
        with self.db.session_scope() as session:
            ids = session.scalars(
                select(ApiErrorEventORM.id)
                .order_by(ApiErrorEventORM.id.desc())
                .offset(self.max_events)
            ).all()
            if not ids:
                return
            session.execute(delete(ApiErrorEventORM).where(ApiErrorEventORM.id.in_(ids)))

    def record_error(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        detail: str,
        role: str | None = None,
    ) -> None:
        with self.db.session_scope() as session:
            session.add(
                ApiErrorEventORM(
                    occurred_at=datetime.now(tz=timezone.utc),
                    request_id=request_id,
                    method=method.upper(),
                    path=path,
                    status_code=int(status_code),
                    detail=detail,
                    role=role if role in {"viewer", "analyst", "admin"} else None,
                )
            )
        self._trim_old_events()

    def recent_errors(self, limit: int = 20) -> list[ApiErrorEventRecord]:
        n = max(1, min(int(limit), 200))
        with self.db.session_scope() as session:
            rows = session.scalars(
                select(ApiErrorEventORM)
                .order_by(ApiErrorEventORM.occurred_at.desc(), ApiErrorEventORM.id.desc())
                .limit(n)
            ).all()
        return [self._to_record(row) for row in rows]

    def summary(self, window_minutes: int = 60) -> dict[str, Any]:
        window = max(1, int(window_minutes))
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(minutes=window)

        with self.db.session_scope() as session:
            total_count = int(session.scalar(select(func.count(ApiErrorEventORM.id))) or 0)
            recent_count = int(
                session.scalar(
                    select(func.count(ApiErrorEventORM.id)).where(
                        ApiErrorEventORM.occurred_at >= cutoff
                    )
                )
                or 0
            )
            status_rows = session.execute(
                select(ApiErrorEventORM.status_code, func.count(ApiErrorEventORM.id))
                .where(ApiErrorEventORM.occurred_at >= cutoff)
                .group_by(ApiErrorEventORM.status_code)
            ).all()
            path_rows = session.execute(
                select(ApiErrorEventORM.path, func.count(ApiErrorEventORM.id))
                .where(ApiErrorEventORM.occurred_at >= cutoff)
                .group_by(ApiErrorEventORM.path)
                .order_by(func.count(ApiErrorEventORM.id).desc())
                .limit(10)
            ).all()
            last_error_at = session.scalar(
                select(ApiErrorEventORM.occurred_at)
                .order_by(ApiErrorEventORM.occurred_at.desc(), ApiErrorEventORM.id.desc())
                .limit(1)
            )

        return {
            "generated_at": now,
            "window_minutes": window,
            "total_error_count": total_count,
            "recent_error_count": recent_count,
            "by_status": {str(status): int(count) for status, count in status_rows},
            "by_path": {str(path): int(count) for path, count in path_rows},
            "last_error_at": last_error_at,
        }
