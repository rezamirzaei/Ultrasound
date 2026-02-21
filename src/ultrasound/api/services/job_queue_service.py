"""Background job queue for model training and dataset ingestion tasks."""

from __future__ import annotations

import logging
import threading
import time
from typing import cast

from ultrasound.api.models.domain import JobRunRecord
from ultrasound.api.models.schemas import BusiTrainingRequest
from ultrasound.api.repositories.job_repository import JobRepository
from ultrasound.api.services.busi_training_service import BusiTrainingService
from ultrasound.api.services.data_ingestion_service import DataIngestionService
from ultrasound.api.services.observability_service import ObservabilityService

logger = logging.getLogger("inphase.jobs")


class JobQueueService:
    """Coordinates asynchronous execution of heavy learning and ingestion workloads."""

    def __init__(
        self,
        repository: JobRepository,
        busi_training_service: BusiTrainingService,
        data_ingestion_service: DataIngestionService,
        observability_service: ObservabilityService,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        self.repository = repository
        self.busi_training_service = busi_training_service
        self.data_ingestion_service = data_ingestion_service
        self.observability_service = observability_service
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))

        self._worker_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start daemon worker thread if not already running."""
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._run_loop,
                name="inphase-job-worker",
                daemon=True,
            )
            self._worker_thread.start()
            self.observability_service.set_worker_up(True)

    def stop(self, timeout_seconds: float = 5.0) -> None:
        """Stop worker thread gracefully."""
        with self._worker_lock:
            thread = self._worker_thread
            self._stop_event.set()
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(0.1, float(timeout_seconds)))
            self._worker_thread = None
            self.observability_service.set_worker_up(False)

    def enqueue_busi_training(
        self, request: BusiTrainingRequest, requested_by: str
    ) -> JobRunRecord:
        return self.repository.enqueue(
            job_type="busi_training",
            requested_by=requested_by,
            payload=request.model_dump(mode="json"),
        )

    def enqueue_dataset_resync(self, requested_by: str) -> JobRunRecord:
        return self.repository.enqueue(
            job_type="dataset_resync",
            requested_by=requested_by,
            payload={"trigger": "manual"},
        )

    def get_job(self, job_id: int) -> JobRunRecord | None:
        return self.repository.get_job(job_id)

    def list_jobs(self, limit: int = 50) -> list[JobRunRecord]:
        return self.repository.list_jobs(limit=limit)

    def _run_loop(self) -> None:
        logger.info("job worker started")
        while not self._stop_event.is_set():
            job = self.repository.claim_next_pending()
            if job is None:
                self._stop_event.wait(self.poll_interval_seconds)
                continue

            self._execute_job(job)

        logger.info("job worker stopped")

    def _execute_job(self, job: JobRunRecord) -> None:
        started = time.perf_counter()
        try:
            if job.job_type == "busi_training":
                request = BusiTrainingRequest.model_validate(job.payload)
                training_result = self.busi_training_service.run_training(request)
                result_payload = {
                    "run_id": training_result.run_id,
                    "train_accuracy": training_result.train_accuracy,
                    "test_accuracy": training_result.test_accuracy,
                    "epochs": training_result.epochs,
                    "train_samples": training_result.train_samples,
                    "test_samples": training_result.test_samples,
                }
            elif job.job_type == "dataset_resync":
                resync_result = self.data_ingestion_service.resync_all()
                result_payload = resync_result.model_dump(mode="json")
            else:
                raise ValueError(f"Unsupported job type '{job.job_type}'.")

            self.repository.mark_completed(job.id, result_payload)
            duration = time.perf_counter() - started
            self.observability_service.observe_job(job.job_type, "completed", duration)
            logger.info(
                "job completed id=%s type=%s duration_ms=%.2f",
                job.id,
                job.job_type,
                duration * 1e3,
            )
        except Exception as exc:
            self.repository.mark_failed(job.id, str(exc))
            duration = time.perf_counter() - started
            job_type = cast(str, job.job_type)
            self.observability_service.observe_job(job_type, "failed", duration)
            logger.exception(
                "job failed id=%s type=%s duration_ms=%.2f error=%s",
                job.id,
                job_type,
                duration * 1e3,
                exc,
            )
