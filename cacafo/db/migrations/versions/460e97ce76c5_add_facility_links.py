"""add facility links

Revision ID: 460e97ce76c5
Revises: 98329c0ff1fa
Create Date: 2024-09-09 14:31:25.990374

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "460e97ce76c5"
down_revision: Union[str, None] = "98329c0ff1fa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "animal_type_annotation", sa.Column("facility_id", sa.Integer(), nullable=True)
    )
    op.add_column(
        "animal_type_annotation",
        sa.Column("annotated_on", sa.DateTime(), nullable=False),
    )
    op.create_foreign_key(
        None, "animal_type_annotation", "facility", ["facility_id"], ["id"]
    )
    op.add_column("building", sa.Column("facility_id", sa.Integer(), nullable=True))
    op.create_foreign_key(None, "building", "facility", ["facility_id"], ["id"])
    op.add_column(
        "cafo_annotation", sa.Column("annotated_on", sa.DateTime(), nullable=False)
    )
    op.add_column(
        "cafo_annotation", sa.Column("facility_id", sa.Integer(), nullable=True)
    )
    op.create_foreign_key(None, "cafo_annotation", "facility", ["facility_id"], ["id"])
    op.add_column(
        "construction_annotation",
        sa.Column("annotated_on", sa.DateTime(), nullable=False),
    )
    op.add_column(
        "construction_annotation", sa.Column("facility_id", sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        None, "construction_annotation", "facility", ["facility_id"], ["id"]
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, "construction_annotation", type_="foreignkey")
    op.drop_column("construction_annotation", "facility_id")
    op.drop_column("construction_annotation", "annotated_on")
    op.drop_constraint(None, "cafo_annotation", type_="foreignkey")
    op.drop_column("cafo_annotation", "facility_id")
    op.drop_column("cafo_annotation", "annotated_on")
    op.drop_constraint(None, "building", type_="foreignkey")
    op.drop_column("building", "facility_id")
    op.drop_constraint(None, "animal_type_annotation", type_="foreignkey")
    op.drop_column("animal_type_annotation", "annotated_on")
    op.drop_column("animal_type_annotation", "facility_id")
    # ### end Alembic commands ###
