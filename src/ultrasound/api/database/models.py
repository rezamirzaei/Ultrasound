"""SQLAlchemy ORM table models."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ultrasound.api.database.session import Base


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class DatasetMetaORM(Base):
    __tablename__ = "dataset_meta"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class BusiSampleORM(Base):
    __tablename__ = "busi_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    class_name: Mapped[str] = mapped_column(String(32), nullable=False)
    image_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    sample_stem: Mapped[str] = mapped_column(String(256), nullable=False)
    image_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    mask_blob: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("ix_busi_samples_class_split", "class_name", "split"),
        Index("ux_busi_samples_class_filename", "class_name", "image_filename", unique=True),
    )


class NdtSampleORM(Base):
    __tablename__ = "ndt_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    rf_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    time_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    n_points: Mapped[int] = mapped_column(Integer, nullable=False)
    fs_hz: Mapped[float] = mapped_column(Float, nullable=False)
    fc_hz: Mapped[float] = mapped_column(Float, nullable=False)
    c_mps: Mapped[float] = mapped_column(Float, nullable=False)
    thickness_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    defects: Mapped[list[NdtDefectORM]] = relationship(
        "NdtDefectORM",
        back_populates="sample",
        cascade="all, delete-orphan",
    )


class NdtDefectORM(Base):
    __tablename__ = "ndt_defects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("ndt_samples.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    depth_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    amplitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    sample: Mapped[NdtSampleORM] = relationship("NdtSampleORM", back_populates="defects")

    __table_args__ = (Index("ix_ndt_defects_sample_ordinal", "sample_id", "ordinal"),)


class BusiTrainingRunORM(Base):
    __tablename__ = "busi_training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    include_normal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    train_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (Index("ix_busi_training_runs_scope", "include_normal", "id"),)


class IndustrialSampleORM(Base):
    __tablename__ = "industrial_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(String(64), nullable=False)
    split: Mapped[str] = mapped_column(String(32), nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    image_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    image_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    annotation_blob: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("ix_industrial_samples_dataset_split", "dataset_name", "split", "class_name"),
        Index(
            "ux_industrial_samples_dataset_path",
            "dataset_name",
            "relative_path",
            unique=True,
        ),
    )


class IndustrialTrainingRunORM(Base):
    __tablename__ = "industrial_training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    dataset_name: Mapped[str] = mapped_column(String(64), nullable=False)
    train_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (Index("ix_industrial_training_runs_dataset_id", "dataset_name", "id"),)


class ApiErrorEventORM(Base):
    __tablename__ = "api_error_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    method: Mapped[str] = mapped_column(String(16), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_api_error_events_occurred_at", "occurred_at"),
        Index("ix_api_error_events_status_path", "status_code", "path"),
    )


class JobRunORM(Base):
    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    requested_by: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_job_runs_status_submitted", "status", "submitted_at"),
        Index("ix_job_runs_type_submitted", "job_type", "submitted_at"),
    )


class AuthUserORM(Base):
    __tablename__ = "auth_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )

    tokens: Mapped[list[AuthTokenORM]] = relationship(
        "AuthTokenORM",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("ix_auth_users_role_active", "role", "is_active"),)


class AuthTokenORM(Base):
    __tablename__ = "auth_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("auth_users.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[AuthUserORM] = relationship("AuthUserORM", back_populates="tokens")

    __table_args__ = (
        Index("ix_auth_tokens_expiry", "expires_at"),
        Index("ix_auth_tokens_user_active", "user_id", "revoked_at"),
    )
