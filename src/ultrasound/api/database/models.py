"""SQLAlchemy ORM table models."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from ultrasound.api.database.session import Base


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class DatasetMetaORM(Base):
    __tablename__ = "dataset_meta"

    key = Column(String(128), primary_key=True)
    value = Column(Text, nullable=False)


class BusiSampleORM(Base):
    __tablename__ = "busi_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    class_name = Column(String(32), nullable=False)
    image_filename = Column(String(256), nullable=False)
    sample_stem = Column(String(256), nullable=False)
    image_blob = Column(LargeBinary, nullable=False)
    mask_blob = Column(LargeBinary, nullable=True)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    label = Column(Integer, nullable=False)
    split = Column(String(16), nullable=False)
    source_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("ix_busi_samples_class_split", "class_name", "split"),
        Index("ux_busi_samples_class_filename", "class_name", "image_filename", unique=True),
    )


class NdtSampleORM(Base):
    __tablename__ = "ndt_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False, unique=True)
    rf_blob = Column(LargeBinary, nullable=False)
    time_blob = Column(LargeBinary, nullable=False)
    n_points = Column(Integer, nullable=False)
    fs_hz = Column(Float, nullable=False)
    fc_hz = Column(Float, nullable=False)
    c_mps = Column(Float, nullable=False)
    thickness_m = Column(Float, nullable=True)
    description = Column(Text, nullable=False)
    source_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    defects = relationship("NdtDefectORM", back_populates="sample", cascade="all, delete-orphan")


class NdtDefectORM(Base):
    __tablename__ = "ndt_defects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(Integer, ForeignKey("ndt_samples.id", ondelete="CASCADE"), nullable=False)
    ordinal = Column(Integer, nullable=False)
    depth_m = Column(Float, nullable=True)
    amplitude = Column(Float, nullable=True)

    sample = relationship("NdtSampleORM", back_populates="defects")

    __table_args__ = (Index("ix_ndt_defects_sample_ordinal", "sample_id", "ordinal"),)


class BusiTrainingRunORM(Base):
    __tablename__ = "busi_training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    include_normal = Column(Boolean, nullable=False, default=False)
    train_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    payload_json = Column(Text, nullable=False)

    __table_args__ = (Index("ix_busi_training_runs_scope", "include_normal", "id"),)


class ApiErrorEventORM(Base):
    __tablename__ = "api_error_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    occurred_at = Column(DateTime(timezone=True), default=utcnow)
    request_id = Column(String(64), nullable=False)
    method = Column(String(16), nullable=False)
    path = Column(String(512), nullable=False)
    status_code = Column(Integer, nullable=False)
    detail = Column(Text, nullable=False)
    role = Column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_api_error_events_occurred_at", "occurred_at"),
        Index("ix_api_error_events_status_path", "status_code", "path"),
    )
