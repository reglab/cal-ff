"""rename model score to bucket

Revision ID: 32813a3035af
Revises: 27bb77c3c7df
Create Date: 2024-09-18 11:45:21.395891

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "32813a3035af"
down_revision: Union[str, None] = "27bb77c3c7df"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("image", sa.Column("bucket", sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("image", "bucket")
    # ### end Alembic commands ###
