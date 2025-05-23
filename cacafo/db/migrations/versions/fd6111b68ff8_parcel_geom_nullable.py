"""parcel geom nullable

Revision ID: fd6111b68ff8
Revises: 310017c371fa
Create Date: 2024-09-10 11:27:19.229049

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry

# revision identifiers, used by Alembic.
revision: str = "fd6111b68ff8"
down_revision: Union[str, None] = "310017c371fa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "parcel",
        "inferred_geometry",
        existing_type=Geometry(
            geometry_type="POLYGON",
            from_text="ST_GeomFromEWKT",
            name="geometry",
            nullable=False,
            _spatial_index_reflected=True,
        ),
        nullable=True,
    )
    op.alter_column(
        "permit",
        "registered_location_parcel_id",
        existing_type=sa.INTEGER(),
        nullable=True,
    )
    op.alter_column(
        "permit",
        "geocoded_address_location_parcel_id",
        existing_type=sa.INTEGER(),
        nullable=True,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "permit",
        "geocoded_address_location_parcel_id",
        existing_type=sa.INTEGER(),
        nullable=False,
    )
    op.alter_column(
        "permit",
        "registered_location_parcel_id",
        existing_type=sa.INTEGER(),
        nullable=False,
    )
    op.alter_column(
        "parcel",
        "inferred_geometry",
        existing_type=Geometry(
            geometry_type="POLYGON",
            from_text="ST_GeomFromEWKT",
            name="geometry",
            nullable=False,
            _spatial_index_reflected=True,
        ),
        nullable=False,
    )
    # ### end Alembic commands ###
