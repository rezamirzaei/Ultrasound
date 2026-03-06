"""Operational diagnostics endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from fastapi import APIRouter, Depends, Query
from sqlalchemy import column, func, inspect, select, table

from ultrasound.api.container import ApplicationContainer
from ultrasound.api.controllers.dependencies import get_container, require_role
from ultrasound.api.database.session import Base
from ultrasound.api.models.schemas import (
    DatabaseSchemaStatusResponse,
    DatabaseTableStatus,
    DatasetResyncResponse,
    OpsErrorEvent,
    OpsErrorSummaryResponse,
)

router = APIRouter(
    tags=["ops"],
    dependencies=[Depends(require_role("admin"))],
)


@router.get("/ops/errors/summary", response_model=OpsErrorSummaryResponse)
def get_error_summary(
    window_minutes: int = Query(default=60, ge=1, le=7 * 24 * 60),
    container: ApplicationContainer = Depends(get_container),
) -> OpsErrorSummaryResponse:
    """Return aggregate API error counters over the requested time window."""
    payload = container.error_analytics_service.summary(window_minutes=window_minutes)
    return OpsErrorSummaryResponse(**payload)


@router.get("/ops/errors/recent", response_model=list[OpsErrorEvent])
def get_recent_errors(
    limit: int = Query(default=20, ge=1, le=200),
    container: ApplicationContainer = Depends(get_container),
) -> list[OpsErrorEvent]:
    """Return most recent API errors for troubleshooting and triage."""
    events = container.error_analytics_service.recent_errors(limit=limit)
    return [OpsErrorEvent(**event.model_dump()) for event in events]


@router.post("/ops/datasets/resync", response_model=DatasetResyncResponse)
def resync_datasets(
    container: ApplicationContainer = Depends(get_container),
) -> DatasetResyncResponse:
    """Resync BUSI, NDT, and industrial source files into DB tables."""
    return container.data_ingestion_service.resync_all()


@router.get("/ops/database/schema-status", response_model=DatabaseSchemaStatusResponse)
def get_database_schema_status(
    container: ApplicationContainer = Depends(get_container),
) -> DatabaseSchemaStatusResponse:
    """Return DB table counts and Alembic revision status for operational checks."""
    with container.db.engine.connect() as connection:
        inspector = inspect(connection)
        existing_tables = set(inspector.get_table_names())

        rows: list[DatabaseTableStatus] = []
        tracked_tables = set(Base.metadata.tables.keys())
        tracked_tables.add("alembic_version")

        for table_name in sorted(tracked_tables):
            if table_name not in existing_tables:
                rows.append(DatabaseTableStatus(table_name=table_name, row_count=0))
                continue

            if table_name == "alembic_version":
                alembic_table = table("alembic_version", column("version_num"))
                row_count = int(
                    connection.scalar(select(func.count()).select_from(alembic_table)) or 0
                )
            else:
                model_table = Base.metadata.tables[table_name]
                row_count = int(
                    connection.scalar(select(func.count()).select_from(model_table)) or 0
                )

            rows.append(DatabaseTableStatus(table_name=table_name, row_count=row_count))

        current_revision: str | None = None
        try:
            migration_context = MigrationContext.configure(connection)
            current_revision = migration_context.get_current_revision()
        except Exception:
            current_revision = None

    head_revision: str | None = None
    try:
        cfg = Config(str(container.config.project_root / "alembic.ini"))
        cfg.set_main_option("script_location", str(container.config.project_root / "alembic"))
        if container.config.database_url:
            cfg.set_main_option("sqlalchemy.url", container.config.database_url)
        script = ScriptDirectory.from_config(cfg)
        head_revision = script.get_current_head()
    except Exception:
        head_revision = None

    return DatabaseSchemaStatusResponse(
        generated_at=datetime.now(tz=timezone.utc),
        database_url=container.config.database_url,
        alembic_current_revision=current_revision,
        alembic_head_revision=head_revision,
        tables=rows,
    )
