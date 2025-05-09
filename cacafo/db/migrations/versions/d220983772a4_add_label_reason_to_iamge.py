"""add label_reason to iamge

Revision ID: d220983772a4
Revises: eb0e00cdb897
Create Date: 2024-10-21 16:44:49.534735

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d220983772a4"
down_revision: Union[str, None] = "eb0e00cdb897"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("image", sa.Column("label_reason", sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("image", "label_reason")
    # ### end Alembic commands ###
