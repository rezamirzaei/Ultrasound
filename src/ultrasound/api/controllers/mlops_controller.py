"""Learning, background jobs, and dataset upload endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.models.domain import AuthSessionRecord, JobRunRecord
from ultrasound.api.models.schemas import (
    BusiTrainingRequest,
    BusiUploadResponse,
    IndustrialUploadResponse,
    JobEnqueueResponse,
    JobRunResponse,
)

router = APIRouter(tags=["mlops"])


def _to_job_response(job: JobRunRecord) -> JobRunResponse:
    return JobRunResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        requested_by=job.requested_by,
        submitted_at=job.submitted_at,
        payload=job.payload,
        result=job.result,
        error_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post("/learning/jobs/busi-training", response_model=JobEnqueueResponse)
def enqueue_busi_training_job(
    request: BusiTrainingRequest,
    current_user: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> JobEnqueueResponse:
    """Queue asynchronous BUSI learning job for production-safe execution."""
    job = container.job_queue_service.enqueue_busi_training(
        request=request,
        requested_by=current_user.username,
    )
    return JobEnqueueResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        requested_by=job.requested_by,
        submitted_at=job.submitted_at,
    )


@router.post("/learning/jobs/datasets-resync", response_model=JobEnqueueResponse)
def enqueue_dataset_resync_job(
    current_user: AuthSessionRecord = Depends(require_role("admin")),
    container: ApplicationContainer = Depends(get_container),
) -> JobEnqueueResponse:
    """Queue asynchronous DB resync for BUSI/NDT/industrial tables."""
    job = container.job_queue_service.enqueue_dataset_resync(requested_by=current_user.username)
    return JobEnqueueResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        requested_by=job.requested_by,
        submitted_at=job.submitted_at,
    )


@router.get("/learning/jobs", response_model=list[JobRunResponse])
def list_learning_jobs(
    limit: int = Query(default=30, ge=1, le=200),
    _role: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> list[JobRunResponse]:
    """List recent training/ingestion background jobs."""
    jobs = container.job_queue_service.list_jobs(limit=limit)
    return [_to_job_response(job) for job in jobs]


@router.get("/learning/jobs/{job_id}", response_model=JobRunResponse)
def get_learning_job(
    job_id: int,
    _role: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> JobRunResponse:
    """Get one background job by id."""
    job = container.job_queue_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _to_job_response(job)


@router.post("/datasets/busi/upload", response_model=BusiUploadResponse)
async def upload_busi_sample(
    class_name: Literal["benign", "malignant", "normal"] = Form(...),
    split: Literal["train", "test"] = Form("train"),
    image: UploadFile = File(...),
    mask: UploadFile | None = File(default=None),
    _role: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> BusiUploadResponse:
    """Upload one BUSI sample into SQL storage with optional mask."""
    image_blob = await image.read()
    mask_blob = await mask.read() if mask is not None else None

    try:
        record = container.dataset_upload_service.upload_busi_sample(
            class_name=class_name,
            split=split,
            image_filename=image.filename or "uploaded_busi.png",
            image_blob=image_blob,
            mask_blob=mask_blob,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BusiUploadResponse(
        sample_id=record.sample_id,
        class_name=record.class_name,
        split=record.split,
        image_filename=record.image_filename,
        total_class_samples=record.total_class_samples,
        storage="sql",
        created_at=record.created_at,
    )


@router.post("/datasets/industrial/upload", response_model=IndustrialUploadResponse)
async def upload_industrial_sample(
    dataset_name: Literal["steel_defect", "neu_surface", "casting_defect"] = Form(...),
    split: str = Form(...),
    class_name: str = Form(...),
    image: UploadFile = File(...),
    annotation: UploadFile | None = File(default=None),
    _role: AuthSessionRecord = Depends(require_role("analyst")),
    container: ApplicationContainer = Depends(get_container),
) -> IndustrialUploadResponse:
    """Upload one industrial sample (steel/NEU/casting) into SQL storage."""
    image_blob = await image.read()
    annotation_blob = await annotation.read() if annotation is not None else None

    try:
        record = container.dataset_upload_service.upload_industrial_sample(
            dataset_name=dataset_name,
            split=split,
            class_name=class_name,
            image_filename=image.filename or "uploaded_industrial.png",
            image_blob=image_blob,
            annotation_blob=annotation_blob,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IndustrialUploadResponse(
        sample_id=record.sample_id,
        dataset_name=record.dataset_name,
        split=record.split,
        class_name=record.class_name,
        image_filename=record.image_filename,
        relative_path=record.relative_path,
        has_annotation=record.has_annotation,
        total_class_samples=record.total_class_samples,
        storage="sql",
        created_at=record.created_at,
    )
