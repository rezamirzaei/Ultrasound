"""Tests for runtime job queue and repository behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from ultrasound.api.database.models import JobRunORM
from ultrasound.api.database.session import DatabaseSessionManager
from ultrasound.api.models.domain import JobRunRecord
from ultrasound.api.models.schemas import (
    BusiTrainingRequest,
    BusiTrainingResponse,
    DatasetResyncResponse,
    IndustrialTrainingRequest,
    IndustrialTrainingResponse,
)
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


class _ThreadStub:
    def __init__(self, *, alive: bool = True) -> None:
        self._alive = alive
        self.started = False
        self.join_calls: list[float] = []

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(float(timeout or 0.0))

    def is_alive(self) -> bool:
        return self._alive


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


def test_job_repository_claims_pending_jobs_and_clamps_list_limit(tmp_path: Path) -> None:
    repository = _job_db(tmp_path)
    first = repository.enqueue("dataset_resync", "admin", {"order": 1})
    second = repository.enqueue("busi_training", "analyst", {"order": 2})

    claimed = repository.claim_next_pending()

    assert claimed is not None
    assert claimed.id == first.id
    assert claimed.status == "running"
    assert repository.get_job(first.id) is not None
    assert repository.get_job(999999) is None
    assert len(repository.list_jobs(limit=0)) == 1
    assert len(repository.list_jobs(limit=500)) == 2
    assert repository.list_jobs(limit=500)[0].id == second.id


def test_job_repository_falls_back_for_invalid_row_payloads(tmp_path: Path) -> None:
    repository = _job_db(tmp_path)

    with repository.db.session_scope() as session:
        session.add(
            JobRunORM(
                job_type="mystery",
                status="unknown",
                requested_by="admin",
                payload_json='["bad"]',
                result_json='"not a dict"',
                error_message=None,
                submitted_at=datetime.now(tz=timezone.utc),
            )
        )

    job = repository.list_jobs(limit=10)[0]

    assert job.job_type == "dataset_resync"
    assert job.status == "failed"
    assert job.payload == {}
    assert job.result is None

    with pytest.raises(RuntimeError, match="no primary key"):
        repository._to_record(  # type: ignore[attr-defined]
            JobRunORM(
                job_type="dataset_resync",
                status="pending",
                requested_by="admin",
                payload_json="{}",
                submitted_at=datetime.now(tz=timezone.utc),
            )
        )


def test_job_repository_missing_mark_targets_raise(tmp_path: Path) -> None:
    repository = _job_db(tmp_path)

    with pytest.raises(ValueError, match="Job 1 not found"):
        repository.mark_completed(1, {"ok": True})
    with pytest.raises(ValueError, match="Job 1 not found"):
        repository.mark_failed(1, "boom")


def test_job_queue_service_delegates_enqueues_and_accessors(monkeypatch) -> None:
    observability = _ObservabilityStub()

    class _RepositoryStub:
        def enqueue(
            self,
            job_type: Literal["busi_training", "dataset_resync", "industrial_training"],
            requested_by: str,
            payload: dict[str, Any],
        ) -> JobRunRecord:
            return JobRunRecord(
                id=1,
                job_type=job_type,
                status="pending",
                requested_by=requested_by,
                payload=payload,
                submitted_at=datetime.now(tz=timezone.utc),
            )

        def get_job(self, job_id: int) -> JobRunRecord | None:
            return JobRunRecord(
                id=job_id,
                job_type="dataset_resync",
                status="pending",
                requested_by="admin",
                payload={"trigger": "manual"},
                submitted_at=datetime.now(tz=timezone.utc),
            )

        def list_jobs(self, limit: int = 50) -> list[JobRunRecord]:
            return [self.get_job(limit)]  # type: ignore[list-item]

        def claim_next_pending(self) -> JobRunRecord | None:
            raise AssertionError("claim_next_pending should not be called")

        def mark_completed(self, job_id: int, result: dict[str, Any]) -> JobRunRecord:
            raise AssertionError("mark_completed should not be called")

        def mark_failed(self, job_id: int, error_message: str) -> JobRunRecord:
            raise AssertionError("mark_failed should not be called")

    service = JobQueueService(
        repository=cast(Any, _RepositoryStub()),
        busi_training_service=_NoopService(),
        industrial_training_service=_NoopService(),
        data_ingestion_service=_NoopService(),
        observability_service=observability,
    )

    busi_job = service.enqueue_busi_training(BusiTrainingRequest(), "analyst")
    resync_job = service.enqueue_dataset_resync("admin")
    industrial_job = service.enqueue_industrial_training(
        IndustrialTrainingRequest(dataset_name="steel_defect"),
        "analyst",
    )

    assert busi_job.job_type == "busi_training"
    assert resync_job.payload == {"trigger": "manual"}
    assert industrial_job.job_type == "industrial_training"
    assert service.get_job(9) is not None
    assert service.list_jobs(limit=7)[0].id == 7


def test_job_queue_service_start_is_idempotent_and_stop_timeout_preserves_worker_up(monkeypatch) -> None:
    observability = _ObservabilityStub()
    created_threads: list[_ThreadStub] = []

    def _make_thread(**kwargs: Any) -> _ThreadStub:
        thread = _ThreadStub(alive=True)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr("ultrasound.api.services.job_queue_service.threading.Thread", _make_thread)

    service = JobQueueService(
        repository=cast(Any, _NoopService()),
        busi_training_service=_NoopService(),
        industrial_training_service=_NoopService(),
        data_ingestion_service=_NoopService(),
        observability_service=observability,
    )

    service.start()
    service.start()

    assert len(created_threads) == 1
    assert created_threads[0].started is True

    service._worker_thread = cast(Any, _ThreadStub(alive=True))
    service.stop(timeout_seconds=0.2)

    assert observability.worker_states[-1] is True
    assert service._worker_thread is not None


def test_job_queue_service_executes_success_paths_and_records_observability(tmp_path: Path) -> None:
    repository = _job_db(tmp_path)
    observability = _ObservabilityStub()

    class _BusiTrainingStub:
        def run_training(self, request: BusiTrainingRequest) -> BusiTrainingResponse:
            return BusiTrainingResponse(
                run_id=11,
                generated_at=datetime.now(tz=timezone.utc),
                include_normal=request.include_normal,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                train_samples=8,
                test_samples=2,
                class_counts={"benign": 4, "malignant": 4},
                class_labels=["benign", "malignant"],
                train_accuracy=0.9,
                test_accuracy=0.8,
                train_loss=0.1,
                test_loss=0.2,
            )

    class _IndustrialTrainingStub:
        def run_training(self, request: IndustrialTrainingRequest) -> IndustrialTrainingResponse:
            return IndustrialTrainingResponse(
                run_id=12,
                generated_at=datetime.now(tz=timezone.utc),
                dataset_name=request.dataset_name,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                train_samples=10,
                test_samples=4,
                class_counts={"crazing": 14},
                class_labels=["crazing"],
                train_accuracy=0.88,
                test_accuracy=0.77,
                train_loss=0.3,
                test_loss=0.4,
                annotated_samples=3,
            )

    class _ResyncStub:
        def resync_all(self) -> DatasetResyncResponse:
            return DatasetResyncResponse(
                generated_at=datetime.now(tz=timezone.utc),
                busi_rows_synced=3,
                ndt_rows_synced=4,
                industrial_rows_synced=5,
            )

    service = JobQueueService(
        repository=repository,
        busi_training_service=_BusiTrainingStub(),
        industrial_training_service=_IndustrialTrainingStub(),
        data_ingestion_service=_ResyncStub(),
        observability_service=observability,
    )

    busi_job = service.enqueue_busi_training(BusiTrainingRequest(), "analyst")
    resync_job = service.enqueue_dataset_resync("admin")
    industrial_job = service.enqueue_industrial_training(
        IndustrialTrainingRequest(dataset_name="steel_defect"),
        "analyst",
    )

    for job_id in (busi_job.id, resync_job.id, industrial_job.id):
        claimed = repository.claim_next_pending()
        assert claimed is not None
        service._execute_job(claimed)
        stored = repository.get_job(job_id)
        assert stored is not None
        assert stored.status == "completed"

    assert observability.jobs == [
        ("busi_training", "completed"),
        ("dataset_resync", "completed"),
        ("industrial_training", "completed"),
    ]


def test_job_queue_service_failure_paths_mark_job_failed_and_record_metric(tmp_path: Path) -> None:
    repository = _job_db(tmp_path)
    observability = _ObservabilityStub()

    class _RepositoryStub(JobRepository):
        def __init__(self, wrapped: JobRepository) -> None:
            self._wrapped = wrapped

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

        def mark_failed(self, job_id: int, error_message: str) -> JobRunRecord:
            raise RuntimeError("cannot persist failure")

    class _BrokenBusiTrainingStub:
        def run_training(self, request: BusiTrainingRequest) -> BusiTrainingResponse:
            raise RuntimeError("training exploded")

    service = JobQueueService(
        repository=_RepositoryStub(repository),
        busi_training_service=_BrokenBusiTrainingStub(),
        industrial_training_service=_NoopService(),
        data_ingestion_service=_NoopService(),
        observability_service=observability,
    )

    repository.enqueue("busi_training", "analyst", BusiTrainingRequest().model_dump(mode="json"))
    claimed = repository.claim_next_pending()
    assert claimed is not None

    service._execute_job(claimed)

    assert observability.jobs == [("busi_training", "failed")]

    unsupported_job = cast(
        JobRunRecord,
        cast(
            Any,
            JobRunRecord(
                id=999,
                job_type="dataset_resync",
                status="running",
                requested_by="admin",
                payload={"trigger": "manual"},
                submitted_at=datetime.now(tz=timezone.utc),
            ),
        ),
    )
    cast(Any, unsupported_job).job_type = "unknown"

    service._execute_job(cast(Any, unsupported_job))

    assert observability.jobs[-1] == ("unknown", "failed")
