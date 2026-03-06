"""Tests for runtime job queue and repository behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import JobRunRecord
from ultrasound.api.repositories.job_repository import JobRepository
from ultrasound.api.services.job_queue_service import JobQueueService


class _ObservabilityStub:
    def __init__(self) -> None:
        self.worker_states: list[bool] = []
        self.jobs: list[tuple[str, str]] = []

    def set_worker_up(self, is_up: bool) -> None:
        self.worker_states.append(is_up)

    def observe_job(self, job_type: str, status: str, duration_seconds: float) -> None:
        self.jobs.append((job_type, status))


class _NoopService:
    def run_training(self, request):  # noqa: ANN001
        raise AssertionError("run_training should not be called")

    def resync_all(self):  # noqa: ANN201
        raise AssertionError("resync_all should not be called")


def _job_db(tmp_path: Path) -> JobRepository:
    db = DatabaseSessionManager(f"sqlite:///{tmp_path / 'jobs.sqlite3'}")
    return JobRepository(db)


def test_job_repository_mark_completed_sets_started_at(tmp_path: Path) -> None:
    repository = _job_db(tmp_path)
    job = repository.enqueue("dataset_resync", "admin", {"trigger": "manual"})

    updated = repository.mark_completed(job.id, {"synced": True})

    assert updated.status == "completed"
    assert updated.started_at is not None
    assert updated.finished_at is not None
    assert updated.result == {"synced": True}


def test_job_repository_mark_failed_clears_stale_result(tmp_path: Path) -> None:
    repository = _job_db(tmp_path)
    job = repository.enqueue("dataset_resync", "admin", {"trigger": "manual"})
    repository.mark_completed(job.id, {"synced": True})

    updated = repository.mark_failed(job.id, "boom")

    assert updated.status == "failed"
    assert updated.result is None
    assert updated.started_at is not None
    assert updated.finished_at is not None
    assert updated.error_message == "boom"


def test_job_worker_loop_survives_polling_errors(monkeypatch) -> None:
    observability = _ObservabilityStub()

    class _RepositoryStub:
        def __init__(self) -> None:
            self.calls = 0

        def enqueue(
            self,
            job_type: Literal["busi_training", "dataset_resync", "industrial_training"],
            requested_by: str,
            payload: dict[str, Any],
        ) -> JobRunRecord:
            raise AssertionError("enqueue should not be called")

        def get_job(self, job_id: int) -> JobRunRecord | None:
            raise AssertionError("get_job should not be called")

        def list_jobs(self, limit: int = 50) -> list[JobRunRecord]:
            raise AssertionError("list_jobs should not be called")

        def claim_next_pending(self) -> JobRunRecord | None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary db failure")
            service._stop_event.set()
            return None

        def mark_completed(self, job_id: int, result: dict[str, Any]) -> JobRunRecord:
            raise AssertionError("mark_completed should not be called")

        def mark_failed(self, job_id: int, error_message: str) -> JobRunRecord:
            raise AssertionError("mark_failed should not be called")

    repository = _RepositoryStub()
    service = JobQueueService(
        repository=repository,
        busi_training_service=_NoopService(),
        industrial_training_service=_NoopService(),
        data_ingestion_service=_NoopService(),
        observability_service=observability,
    )
    monkeypatch.setattr(service._stop_event, "wait", lambda _timeout=None: None)

    service._run_loop()

    assert repository.calls == 2
    assert observability.worker_states[-1] is False
