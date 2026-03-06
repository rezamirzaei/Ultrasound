"""add industrial training runs table

Revision ID: 20260221_0002
Revises: 20260221_0001
Create Date: 2026-02-21 00:40:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260221_0002"
down_revision = "20260221_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "industrial_training_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dataset_name", sa.String(length=64), nullable=False),
        sa.Column("train_accuracy", sa.Float(), nullable=True),
        sa.Column("test_accuracy", sa.Float(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_industrial_training_runs_dataset_id",
        "industrial_training_runs",
        ["dataset_name", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_industrial_training_runs_dataset_id", table_name="industrial_training_runs")
    op.drop_table("industrial_training_runs")
