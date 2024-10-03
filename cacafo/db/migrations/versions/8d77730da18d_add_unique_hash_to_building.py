"""add unique hash to building


Revision ID: 8d77730da18d
Revises: 2cb7464cdb45
Create Date: 2024-10-03 14:34:02.245377

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8d77730da18d"
down_revision: Union[str, None] = "2cb7464cdb45"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("building", sa.Column("hash", sa.String(), nullable=False))
    op.create_unique_constraint(None, "building", ["hash"])
    op.add_column("permit", sa.Column("facility_id", sa.Integer(), nullable=True))
    op.create_foreign_key(None, "permit", "facility", ["facility_id"], ["id"])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, "permit", type_="foreignkey")
    op.drop_column("permit", "facility_id")
    op.drop_constraint(None, "building", type_="unique")
    op.drop_column("building", "hash")
    # ### end Alembic commands ###
