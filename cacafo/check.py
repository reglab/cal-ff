import geoalchemy2 as ga
import rich
import rich_click as click
import sqlalchemy as sa
from shapely import STRtree

import cacafo.db.sa_models as m
from cacafo.db.session import get_sqlalchemy_session

checks = {}


def check(expected=None):
    def wrapper(func):
        checks[func] = expected

    return wrapper


@check(expected=0)
def facilities_with_overlapping_bounding_boxes():
    session = get_sqlalchemy_session()
    query = sa.select(m.Facility.id, m.Facility.geometry).where(
        m.Facility.archived_at.is_(None)
    )
    facilities = session.execute(query).all()

    # Create STRtree
    tree = STRtree([ga.shape.to_shape(f.geometry).envelope for f in facilities])

    results = []
    for idx, facility in enumerate(facilities):
        intersecting = tree.query(ga.shape.to_shape(facility.geometry).envelope)
        for other_idx in intersecting:
            if idx != other_idx:
                results.append((facility.id, facilities[other_idx].id))

    for facility_id, other_facility_id in results:
        rich.print(
            f"[yellow]{facility_id}'s bounding box intersects with {other_facility_id}'s bounding box[/yellow]"
        )

    return len(results)


@click.command()
def check():
    for func, expected in checks.items():
        result = func()
        name = func.__name__.replace("_", " ")
        if expected is not None and result != expected:
            rich.print(f"[red]Error[/red]: {name}: {result} != {expected}")
        else:
            rich.print(f"[green]OK[/green] {name}: {result}")
