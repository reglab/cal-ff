import geoalchemy2 as ga
import rich_click as click
import sqlalchemy as sa
from shapely import STRtree

import cacafo.db.models as m
from cacafo.db.session import new_session


@click.group("urban-mask", help="Commands for managing the urban mask")
def _cli():
    pass


@_cli.command("apply", help="Apply the urban mask to images in the db")
def apply():
    session = new_session()
    urban_mask = session.execute(sa.select(m.UrbanMask)).scalars().all()
    unremoved_images = (
        session.execute(sa.select(m.Image).where(m.Image.bucket.is_not(None)))
        .scalars()
        .all()
    )
    urban_mask_geoms = [ga.shape.to_shape(um.geometry) for um in urban_mask]
    urban_mask_tree = STRtree(urban_mask_geoms)
    unremoved_image_geoms = [ga.shape.to_shape(ui.geometry) for ui in unremoved_images]
    unremoved_image_idxs, urban_mask_idxs = urban_mask_tree.query(
        unremoved_image_geoms, predicate="within"
    )
    unremoved_image_ids = [unremoved_images[i].id for i in unremoved_image_idxs]
    print(f"Found {len(unremoved_image_ids)} images within urban mask")
    session.execute(
        sa.update(m.Image)
        .where(m.Image.id.in_(unremoved_image_ids))
        .values(bucket=None)
    )
    session.commit()
    session.close()
