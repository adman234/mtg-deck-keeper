"""deck type

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-27 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('scan_jobs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('deck_type', sa.String(length=20), server_default='other', nullable=False))


def downgrade() -> None:
    with op.batch_alter_table('scan_jobs', schema=None) as batch_op:
        batch_op.drop_column('deck_type')
