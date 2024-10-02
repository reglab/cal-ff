"""add archived_at field

Revision ID: df12e8368dc7
Revises: db5b12970288
Create Date: 2024-10-01 17:15:53.708942

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "df12e8368dc7"
down_revision: Union[str, None] = "db5b12970288"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("facility", sa.Column("archived_at", sa.DateTime(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("facility", "archived_at")
    # ### end Alembic commands ###
