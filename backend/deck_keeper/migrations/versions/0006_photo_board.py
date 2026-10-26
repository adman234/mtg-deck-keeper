"""photo board

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-26 17:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('scan_images', schema=None) as batch_op:
        batch_op.add_column(sa.Column('board', sa.String(length=20), server_default='main', nullable=False))


def downgrade() -> None:
    with op.batch_alter_table('scan_images', schema=None) as batch_op:
        batch_op.drop_column('board')
