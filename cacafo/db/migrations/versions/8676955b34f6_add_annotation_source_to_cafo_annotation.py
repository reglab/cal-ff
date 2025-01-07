"""Add annotation source to cafo_annotation

Revision ID: 8676955b34f6
Revises: 72a2552e913f
Create Date: 2025-01-06 16:16:01.210225

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8676955b34f6"
down_revision: Union[str, None] = "72a2552e913f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "cafo_annotation", sa.Column("annotation_phase", sa.String(), nullable=False)
    )


def downgrade() -> None:
    op.drop_column("cafo_annotation", "annotation_phase")
