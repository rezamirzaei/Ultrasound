"""baseline schema for API ORM tables

Revision ID: 20260221_0001
Revises:
Create Date: 2026-02-21 00:01:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260221_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_meta",
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )

    op.create_table(
        "busi_samples",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("class_name", sa.String(length=32), nullable=False),
        sa.Column("image_filename", sa.String(length=256), nullable=False),
        sa.Column("sample_stem", sa.String(length=256), nullable=False),
        sa.Column("image_blob", sa.LargeBinary(), nullable=False),
        sa.Column("mask_blob", sa.LargeBinary(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("label", sa.Integer(), nullable=False),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("class_name", "image_filename", name="ux_busi_samples_class_filename"),
    )
    op.create_index(
        "ix_busi_samples_class_split",
        "busi_samples",
        ["class_name", "split"],
        unique=False,
    )

    op.create_table(
        "ndt_samples",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("rf_blob", sa.LargeBinary(), nullable=False),
        sa.Column("time_blob", sa.LargeBinary(), nullable=False),
        sa.Column("n_points", sa.Integer(), nullable=False),
        sa.Column("fs_hz", sa.Float(), nullable=False),
        sa.Column("fc_hz", sa.Float(), nullable=False),
        sa.Column("c_mps", sa.Float(), nullable=False),
        sa.Column("thickness_m", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "ndt_defects",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("sample_id", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("depth_m", sa.Float(), nullable=True),
        sa.Column("amplitude", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["sample_id"], ["ndt_samples.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_ndt_defects_sample_ordinal",
        "ndt_defects",
        ["sample_id", "ordinal"],
        unique=False,
    )

    op.create_table(
        "busi_training_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("include_normal", sa.Boolean(), nullable=False),
        sa.Column("train_accuracy", sa.Float(), nullable=True),
        sa.Column("test_accuracy", sa.Float(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_busi_training_runs_scope",
        "busi_training_runs",
        ["include_normal", "id"],
        unique=False,
    )

    op.create_table(
        "industrial_samples",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dataset_name", sa.String(length=64), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.Column("class_name", sa.String(length=64), nullable=False),
        sa.Column("image_filename", sa.String(length=256), nullable=False),
        sa.Column("relative_path", sa.String(length=1024), nullable=False),
        sa.Column("image_blob", sa.LargeBinary(), nullable=False),
        sa.Column("annotation_blob", sa.LargeBinary(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_name", "relative_path", name="ux_industrial_samples_dataset_path"),
    )
    op.create_index(
        "ix_industrial_samples_dataset_split",
        "industrial_samples",
        ["dataset_name", "split", "class_name"],
        unique=False,
    )

    op.create_table(
        "api_error_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("method", sa.String(length=16), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_api_error_events_occurred_at",
        "api_error_events",
        ["occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_api_error_events_status_path",
        "api_error_events",
        ["status_code", "path"],
        unique=False,
    )

    op.create_table(
        "job_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_by", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_job_runs_status_submitted",
        "job_runs",
        ["status", "submitted_at"],
        unique=False,
    )
    op.create_index(
        "ix_job_runs_type_submitted",
        "job_runs",
        ["job_type", "submitted_at"],
        unique=False,
    )

    op.create_table(
        "auth_users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )
    op.create_index(
        "ix_auth_users_role_active",
        "auth_users",
        ["role", "is_active"],
        unique=False,
    )

    op.create_table(
        "auth_tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["auth_users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_auth_tokens_expiry", "auth_tokens", ["expires_at"], unique=False)
    op.create_index(
        "ix_auth_tokens_user_active",
        "auth_tokens",
        ["user_id", "revoked_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_auth_tokens_user_active", table_name="auth_tokens")
    op.drop_index("ix_auth_tokens_expiry", table_name="auth_tokens")
    op.drop_table("auth_tokens")

    op.drop_index("ix_auth_users_role_active", table_name="auth_users")
    op.drop_table("auth_users")

    op.drop_index("ix_job_runs_type_submitted", table_name="job_runs")
    op.drop_index("ix_job_runs_status_submitted", table_name="job_runs")
    op.drop_table("job_runs")

    op.drop_index("ix_api_error_events_status_path", table_name="api_error_events")
    op.drop_index("ix_api_error_events_occurred_at", table_name="api_error_events")
    op.drop_table("api_error_events")

    op.drop_index("ix_industrial_samples_dataset_split", table_name="industrial_samples")
    op.drop_table("industrial_samples")

    op.drop_index("ix_busi_training_runs_scope", table_name="busi_training_runs")
    op.drop_table("busi_training_runs")

    op.drop_index("ix_ndt_defects_sample_ordinal", table_name="ndt_defects")
    op.drop_table("ndt_defects")
    op.drop_table("ndt_samples")

    op.drop_index("ix_busi_samples_class_split", table_name="busi_samples")
    op.drop_table("busi_samples")

    op.drop_table("dataset_meta")
