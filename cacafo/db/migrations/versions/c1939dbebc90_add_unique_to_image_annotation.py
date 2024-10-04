"""add unique to image annotation

Revision ID: c1939dbebc90
Revises: 8d77730da18d
Create Date: 2024-10-03 16:45:20.340005

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1939dbebc90"
down_revision: Union[str, None] = "8d77730da18d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_unique_constraint(None, "image_annotation", ["annotated_at", "image_id"])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, "image_annotation", type_="unique")
    # ### end Alembic commands ###
