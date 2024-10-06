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
def facilities_with_overlapping_bounding_boxes(verbose=False):
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
        if verbose:
            rich.print(
                f"[yellow]{facility_id}'s bounding box intersects with {other_facility_id}'s bounding box[/yellow]"
            )

    return len(results)


@check(expected=0)
def overlapping_parcels(verbose=False):
    session = get_sqlalchemy_session()
    query = sa.select(
        m.Parcel.id,
        m.Parcel.inferred_geometry,
        sa.func.ST_Area(m.Parcel.inferred_geometry).label("area_m2"),
    ).where(m.Parcel.inferred_geometry.is_not(None))
    results = list(session.execute(query).all())

    # Create STRtree
    parcel_geometries = [ga.shape.to_shape(r[1]).envelope for r in results]
    tree = STRtree(parcel_geometries)
    idx_1, idx_2 = tree.query(parcel_geometries, predicate="intersects")
    intersections = [
        (results[i][0], results[j][0]) for i, j in zip(idx_1, idx_2) if i != j
    ]

    parcel_id_to_area = {r[0]: r[2] for r in results}

    for parcel_id, other_parcel_id in intersections:
        if verbose:
            rich.print(
                f"Parcel {parcel_id} area: {parcel_id_to_area[parcel_id] / 1_000_000:.6f} km²"
            )
            rich.print(
                f"[yellow]{parcel_id} intersects with {other_parcel_id}[/yellow]"
            )
    return len(intersections)


@check(expected=0)
def unmatched_cafo_annotations(verbose=False):
    session = get_sqlalchemy_session()
    query = sa.select(m.CafoAnnotation.id).where(m.CafoAnnotation.facility_id.is_(None))
    results = list(session.execute(query).all())
    for result in results:
        if verbose:
            rich.print(
                f"[yellow]CafoAnnotation {result[0]} is not matched to a facility[/yellow]"
            )
    return len(results)


@check(expected=0)
def unmatched_animal_type_annotations(verbose=False):
    session = get_sqlalchemy_session()
    query = sa.select(m.AnimalTypeAnnotation.id).where(
        m.AnimalTypeAnnotation.facility_id.is_(None)
    )
    results = list(session.execute(query).all())
    for result in results:
        if verbose:
            rich.print(
                f"[yellow]AnimalTypeAnnotation {result[0]} is not matched to a facility[/yellow]"
            )
    return len(results)


@check(expected=0)
def unmatched_construction_annotations(verbose=False):
    session = get_sqlalchemy_session()
    query = sa.select(m.ConstructionAnnotation.id).where(
        m.ConstructionAnnotation.facility_id.is_(None)
    )
    results = list(session.execute(query).all())
    for result in results:
        if verbose:
            rich.print(
                f"[yellow]ConstructionAnnotation {result[0]} is not matched to a facility[/yellow]"
            )
    return len(results)


@check(expected=0)
def facilities_with_no_cafo_annotations(verbose=False):
    session = get_sqlalchemy_session()
    query = (
        sa.select(m.Facility.id)
        .join(
            m.CafoAnnotation,
            m.Facility.id == m.CafoAnnotation.facility_id,
            isouter=True,
        )
        .where(m.Facility.archived_at.is_(None) & m.CafoAnnotation.id.is_(None))
    )
    results = list(session.execute(query).all())
    for result in results:
        if verbose:
            rich.print(
                f"[yellow]Facility {result[0]} has no matched CafoAnnotation[/yellow]"
            )
    return len(results)


@check(expected=0)
def facilities_with_no_construction_annotations(verbose=False):
    session = get_sqlalchemy_session()
    query = (
        sa.select(m.Facility.id)
        .join(
            m.ConstructionAnnotation,
            m.Facility.id == m.ConstructionAnnotation.facility_id,
            isouter=True,
        )
        .where(m.Facility.archived_at.is_(None) & m.ConstructionAnnotation.id.is_(None))
    )
    results = list(session.execute(query).all())
    for result in results:
        if verbose:
            rich.print(
                f"[yellow]Facility {result[0]} has no matched ConstructionAnnotation[/yellow]"
            )
    return len(results)


@check(expected=0)
def facilities_with_no_animal_type(verbose=False):
    session = get_sqlalchemy_session()
    query = (
        sa.select(m.Facility.id)
        .join(
            m.AnimalTypeAnnotation,
            m.Facility.id == m.AnimalTypeAnnotation.facility_id,
            isouter=True,
        )
        .join(m.Permit, m.Facility.id == m.Permit.facility_id, isouter=True)
        .where(
            m.Facility.archived_at.is_(None)
            & m.AnimalTypeAnnotation.id.is_(None)
            & m.Permit.id.is_(None)
        )
    )
    results = list(session.execute(query).all())
    for result in results:
        if verbose:
            rich.print(
                f"[yellow]Facility {result[0]} has no matched AnimalTypeAnnotation[/yellow]"
            )
    return len(results)


@click.command("check", help="Run data validation checks.")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Print more information about the checks.",
)
@click.option(
    "--check",
    "-c",
    multiple=True,
    help="Run only the specified checks.",
    type=click.Choice([c.__name__ for c in checks.keys()]),
)
def _cli(verbose, check):
    if check:
        checks_to_run = {k: v for k, v in checks.items() if k.__name__ in check}
    else:
        checks_to_run = checks
    for func, expected in checks_to_run.items():
        result = func(verbose)
        name = func.__name__.replace("_", " ")
        if expected is not None and result != expected:
            rich.print(f"[red]Failure[/red]: {name}: {result} != {expected}")
        else:
            rich.print(f"[green]OK[/green] {name}: {result}")
