"""Repository for background job queue persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, cast

from sqlalchemy import select

from ultrasound.api.database.models import JobRunORM
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import JobRunRecord


class JobRepository:
    """Persists and updates asynchronous job records."""

    def __init__(self, db: DatabaseSessionManager):
        self.db = db
        self.db.create_schema()

    def _parse_json_dict(self, raw: str | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _to_record(self, row: JobRunORM) -> JobRunRecord:
        if row.id is None:
            raise RuntimeError("Job row has no primary key")

        payload = self._parse_json_dict(row.payload_json) or {}
        result = self._parse_json_dict(row.result_json)
        status = str(row.status)
        job_type = str(row.job_type)

        if status not in {"pending", "running", "completed", "failed"}:
            status = "failed"
        if job_type not in {"busi_training", "dataset_resync"}:
            job_type = "dataset_resync"

        submitted_at = row.submitted_at or datetime.now(tz=timezone.utc)

        return JobRunRecord(
            id=int(row.id),
            job_type=cast(Literal["busi_training", "dataset_resync"], job_type),
            status=cast(Literal["pending", "running", "completed", "failed"], status),
            requested_by=str(row.requested_by),
            payload=payload,
            result=result,
            error_message=row.error_message,
            submitted_at=submitted_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )

    def enqueue(
        self,
        job_type: Literal["busi_training", "dataset_resync"],
        requested_by: str,
        payload: dict[str, Any],
    ) -> JobRunRecord:
        with self.db.session_scope() as session:
            row = JobRunORM(
                job_type=job_type,
                status="pending",
                requested_by=requested_by,
                payload_json=json.dumps(payload, sort_keys=True),
                submitted_at=datetime.now(tz=timezone.utc),
            )
            session.add(row)
            session.flush()
            if row.id is None:
                raise RuntimeError("Could not persist job")
            return self._to_record(row)

    def claim_next_pending(self) -> JobRunRecord | None:
        with self.db.session_scope() as session:
            row = session.scalars(
                select(JobRunORM)
                .where(JobRunORM.status == "pending")
                .order_by(JobRunORM.submitted_at.asc(), JobRunORM.id.asc())
                .limit(1)
            ).first()
            if row is None:
                return None

            row.status = "running"
            row.started_at = datetime.now(tz=timezone.utc)
            session.flush()
            return self._to_record(row)

    def mark_completed(self, job_id: int, result: dict[str, Any]) -> JobRunRecord:
        with self.db.session_scope() as session:
            row = session.get(JobRunORM, int(job_id))
            if row is None:
                raise ValueError(f"Job {job_id} not found")

            row.status = "completed"
            row.result_json = json.dumps(result, sort_keys=True)
            row.error_message = None
            row.finished_at = datetime.now(tz=timezone.utc)
            session.flush()
            return self._to_record(row)

    def mark_failed(self, job_id: int, error_message: str) -> JobRunRecord:
        with self.db.session_scope() as session:
            row = session.get(JobRunORM, int(job_id))
            if row is None:
                raise ValueError(f"Job {job_id} not found")

            row.status = "failed"
            row.error_message = error_message[:4000]
            row.finished_at = datetime.now(tz=timezone.utc)
            session.flush()
            return self._to_record(row)

    def get_job(self, job_id: int) -> JobRunRecord | None:
        with self.db.session_scope() as session:
            row = session.get(JobRunORM, int(job_id))
            if row is None:
                return None
            return self._to_record(row)

    def list_jobs(self, limit: int = 50) -> list[JobRunRecord]:
        n = max(1, min(int(limit), 200))
        with self.db.session_scope() as session:
            rows = session.scalars(
                select(JobRunORM)
                .order_by(JobRunORM.submitted_at.desc(), JobRunORM.id.desc())
                .limit(n)
            ).all()
        return [self._to_record(row) for row in rows]
