"""create urban mask table

Revision ID: 6a14adc49f24
Revises: 9ff85ca9c925
Create Date: 2024-10-07 16:01:26.938771

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry

# revision identifiers, used by Alembic.
revision: str = "6a14adc49f24"
down_revision: Union[str, None] = "9ff85ca9c925"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_geospatial_table(
        "urban_mask",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uace", sa.Integer(), nullable=False),
        sa.Column("geoid", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("namelsad", sa.String(), nullable=False),
        sa.Column("lsad", sa.String(), nullable=False),
        sa.Column("mtfcc", sa.String(), nullable=False),
        sa.Column("uatyp", sa.String(), nullable=False),
        sa.Column("funcstat", sa.String(), nullable=False),
        sa.Column("aland", sa.BigInteger(), nullable=False),
        sa.Column("awater", sa.BigInteger(), nullable=False),
        sa.Column("intptlat", sa.String(), nullable=False),
        sa.Column("intptlon", sa.String(), nullable=False),
        sa.Column(
            "geometry",
            Geometry(
                geometry_type="MULTIPOLYGON",
                srid=4326,
                spatial_index=False,
                from_text="ST_GeomFromEWKT",
                name="geometry",
                nullable=False,
            ),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_geospatial_index(
        "idx_urban_mask_geometry",
        "urban_mask",
        ["geometry"],
        unique=False,
        postgresql_using="gist",
        postgresql_ops={},
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_geospatial_index(
        "idx_urban_mask_geometry",
        table_name="urban_mask",
        postgresql_using="gist",
        column_name="geometry",
    )
    op.drop_geospatial_table("urban_mask")
    # ### end Alembic commands ###
