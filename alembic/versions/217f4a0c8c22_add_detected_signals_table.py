"""add_detected_signals_table

Revision ID: 217f4a0c8c22
Revises: 0b74ed20d744
Create Date: 2026-04-23 04:38:54.967825

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '217f4a0c8c22'
down_revision: Union[str, Sequence[str], None] = '0b74ed20d744'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'detected_signals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('signal_type', sa.String(length=32), nullable=False),
        sa.Column('severity', sa.String(length=8), nullable=False),
        sa.Column('entity_name', sa.String(length=256), nullable=True, server_default=''),
        sa.Column('entity_type', sa.String(length=64), nullable=True, server_default=''),
        sa.Column('cluster_id', sa.String(length=64), nullable=True, server_default=''),
        sa.Column('cluster_label', sa.String(length=256), nullable=True, server_default=''),
        sa.Column('domain', sa.String(length=64), nullable=True, server_default=''),
        sa.Column('spike_ratio', sa.Float(), nullable=False),
        sa.Column('current_count', sa.Integer(), nullable=False),
        sa.Column('baseline_count', sa.Float(), nullable=False),
        sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('signal_type', 'entity_name', 'cluster_id', 'detected_at',
                            name='uq_signal_subject_detected_at'),
    )
    op.create_index('ix_detected_signals_signal_type', 'detected_signals', ['signal_type'])
    op.create_index('ix_detected_signals_detected_at', 'detected_signals', ['detected_at'])
    op.create_index('ix_detected_signals_expires_at', 'detected_signals', ['expires_at'])


def downgrade() -> None:
    op.drop_table('detected_signals')
