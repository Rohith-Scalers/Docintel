"""Initial schema — documents, regions, chunks tables.

Revision ID: 0001
Revises:
Create Date: 2026-04-07
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create documents, regions, and chunks tables with all indexes.

    pgvector embedding column is NOT created here — the PostgresAdapter adds
    it at runtime with ALTER TABLE ... ADD COLUMN IF NOT EXISTS once the
    extension and configured dimensions are known.

    Returns:
        None
    """
    op.create_table(
        "documents",
        sa.Column("document_id", sa.Text, primary_key=True),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("content_hash", sa.Text, nullable=False, unique=True),
        sa.Column("format", sa.Text, nullable=False),
        sa.Column("total_pages", sa.Integer, nullable=False, server_default="0"),
        sa.Column("processing_status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default="2"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"], unique=True)

    op.create_table(
        "regions",
        sa.Column("region_id", sa.Text, primary_key=True),
        sa.Column(
            "document_id",
            sa.Text,
            sa.ForeignKey("documents.document_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("region_index", sa.Integer, nullable=False),
        sa.Column("region_type", sa.Text, nullable=False),
        sa.Column("bbox", JSONB, nullable=False, server_default="{}"),
        sa.Column("cropped_image_path", sa.Text, nullable=False, server_default=""),
        sa.Column("content_hash", sa.Text, nullable=False, server_default=""),
        sa.Column("detector_confidence", sa.Float, nullable=False, server_default="0"),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default="2"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_regions_document_id", "regions", ["document_id"])

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.Text, primary_key=True),
        sa.Column(
            "document_id",
            sa.Text,
            sa.ForeignKey("documents.document_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("reading_order_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("content_type", sa.Text, nullable=False),
        sa.Column("raw_text", sa.Text, nullable=False, server_default=""),
        sa.Column("overview", sa.Text, nullable=False, server_default=""),
        sa.Column("table_data", JSONB, nullable=True),
        sa.Column("image_description", sa.Text, nullable=True),
        sa.Column("formula_latex", sa.Text, nullable=True),
        sa.Column("structured_data", JSONB, nullable=False, server_default="{}"),
        sa.Column("entities", JSONB, nullable=False, server_default="[]"),
        sa.Column("cropped_image_path", sa.Text, nullable=False, server_default=""),
        sa.Column("confidence_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("correction_applied", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("page_break_context", sa.Text, nullable=True),
        sa.Column("caption", sa.Text, nullable=True),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default="2"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_content_type", "chunks", ["content_type"])
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_structured_data "
        "ON chunks USING GIN (structured_data)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_table_data "
        "ON chunks USING GIN (table_data) WHERE table_data IS NOT NULL"
    )


def downgrade() -> None:
    """Drop all tables created by this migration.

    Returns:
        None
    """
    op.drop_table("chunks")
    op.drop_table("regions")
    op.drop_table("documents")
