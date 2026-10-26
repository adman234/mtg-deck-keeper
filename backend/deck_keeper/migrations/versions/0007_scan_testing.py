"""scan testing

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-27 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('scan_jobs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('expected', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('score_history', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('scan_jobs', schema=None) as batch_op:
        batch_op.drop_column('score_history')
        batch_op.drop_column('expected')
