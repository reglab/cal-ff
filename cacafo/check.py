import inspect

import geoalchemy2 as ga
import rich
import rich_click as click
import sqlalchemy as sa
from shapely import STRtree

import cacafo.db.sa_models as m
from cacafo.db.session import get_sqlalchemy_session
from cacafo.transform import to_meters

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
def cafos_with_no_construction_annotations(verbose=False):
    session = get_sqlalchemy_session()
    query = (
        sa.select(m.Facility)
        .options(sa.orm.joinedload(m.Facility.all_cafo_annotations))
        .join(
            m.ConstructionAnnotation,
            m.Facility.id == m.ConstructionAnnotation.facility_id,
            isouter=True,
        )
        .where(m.Facility.archived_at.is_(None) & m.ConstructionAnnotation.id.is_(None))
    )
    results = list(session.execute(query).scalars().unique().all())
    nca = []
    for facility in results:
        # if facility.is_cafo and not facility.all_construction_annotations:
        if facility.is_cafo and not facility.all_construction_annotations:
            if verbose:
                rich.print(
                    f"[yellow]Facility {facility.id} has no matched ConstructionAnnotation[/yellow]"
                )
            nca.append(facility)
    return len(nca)


@check(expected=0)
def cafos_with_no_animal_type(verbose=False):
    session = get_sqlalchemy_session()
    facilities = (
        session.execute(
            sa.select(m.Facility)
            .options(
                sa.orm.joinedload(m.Facility.all_cafo_annotations),
            )
            .options(
                sa.orm.joinedload(m.Facility.all_animal_type_annotations),
            )
            .options(
                sa.orm.joinedload(m.Facility.best_permits),
            )
            .where(m.Facility.archived_at.is_(None))
        )
        .unique()
        .scalars()
        .all()
    )
    for facility in facilities:
        if facility.is_cafo and not facility.animal_types:
            if verbose:
                rich.print(
                    f"[yellow]Facility {facility.id} has no AnimalTypeAnnotations[/yellow]"
                )
    return len([f for f in facilities if not f.animal_types])


@check(expected=lambda value: value > 2200 and value < 2500)
def facilities_that_are_cafos(verbose=False):
    session = get_sqlalchemy_session()
    facilities = (
        session.execute(
            sa.select(m.Facility)
            .options(
                sa.orm.joinedload(m.Facility.best_permits),
            )
            .options(
                sa.orm.joinedload(m.Facility.all_cafo_annotations),
            )
            .where(m.Facility.archived_at.is_(None))
        )
        .unique()
        .scalars()
        .all()
    )
    cafos = [f for f in facilities if f.is_cafo]
    return len(cafos)


@check(expected=lambda value: value < 700 and value > 400)
def permits_with_no_close_facility(verbose=False):
    session = get_sqlalchemy_session()
    # get permits more than 1km from any cafo
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()

    facilities = (
        session.execute(
            sa.select(m.Facility)
            .options(
                sa.orm.joinedload(m.Facility.best_permits),
            )
            .options(
                sa.orm.joinedload(m.Facility.all_cafo_annotations),
            )
            .where(m.Facility.archived_at.is_(None))
        )
        .unique()
        .scalars()
        .all()
    )

    facilities_tree = STRtree(
        [to_meters(ga.shape.to_shape(f.geometry)) for f in facilities]
    )
    # get permits more than 1km from any cafo
    permits_with_no_close_facility = []
    for permit in permits:
        if permit.facility_id:
            continue
        registered = permit.registered_location and to_meters(
            ga.shape.to_shape(permit.registered_location)
        )
        geocoded = permit.geocoded_address_location and to_meters(
            ga.shape.to_shape(permit.geocoded_address_location)
        )
        registered_close_facilities = facilities_tree.query(
            registered, predicate="dwithin", distance=1000
        )
        geocoded_close_facilities = facilities_tree.query(
            geocoded, predicate="dwithin", distance=1000
        )
        if not len(registered_close_facilities) and not len(geocoded_close_facilities):
            permits_with_no_close_facility.append(permit)
            if verbose:
                rich.print(
                    f"[yellow]Permit {permit.id} (WDID: {permit.data['WDID']}) with animal count {permit.data['Cafo Population']} and termination date {permit.data['Termination Date']} is more than 1km from any facility[/yellow]"
                )
    return len(permits_with_no_close_facility)


@check(expected=lambda value: value > 100)
def large_active_permits_with_no_close_facility(verbose=False):
    session = get_sqlalchemy_session()
    # get permits more than 1km from any cafo
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()

    facilities = (
        session.execute(
            sa.select(m.Facility)
            .options(
                sa.orm.joinedload(m.Facility.best_permits),
            )
            .options(
                sa.orm.joinedload(m.Facility.all_cafo_annotations),
            )
            .where(m.Facility.archived_at.is_(None))
        )
        .unique()
        .scalars()
        .all()
    )

    facilities_tree = STRtree(
        [to_meters(ga.shape.to_shape(f.geometry)) for f in facilities]
    )
    # get permits more than 1km from any cafo
    permits_with_no_close_facility = []
    for permit in permits:
        if permit.facility_id:
            continue
        registered = permit.registered_location and to_meters(
            ga.shape.to_shape(permit.registered_location)
        )
        geocoded = permit.geocoded_address_location and to_meters(
            ga.shape.to_shape(permit.geocoded_address_location)
        )
        registered_close_facilities = facilities_tree.query(
            registered, predicate="dwithin", distance=1000
        )
        geocoded_close_facilities = facilities_tree.query(
            geocoded, predicate="dwithin", distance=1000
        )
        if (
            not len(registered_close_facilities)
            and not len(geocoded_close_facilities)
            and permit.data["Cafo Population"]
            and float(permit.data["Cafo Population"]) > 200
            and not permit.data["Termination Date"]
        ):
            permits_with_no_close_facility.append(permit)
            if verbose:
                rich.print(
                    f"[yellow]Permit {permit.id} (WDID: {permit.data['WDID']}) with animal count {permit.data['Cafo Population']} and termination date {permit.data['Termination Date']} is more than 1km from any facility[/yellow]"
                )
    return len(permits_with_no_close_facility)


@check(expected=0)
def unlabeled_adjacent_images(verbose=False):
    session = get_sqlalchemy_session()

    unlabeled_images = (
        session.execute(
            sa.select(m.Image)
            .join(m.ImageAnnotation, isouter=True)
            .where((m.ImageAnnotation.id.is_(None)) & (m.Image.bucket.is_not(None)))
        )
        .unique()
        .scalars()
        .all()
    )
    facility_images = (
        session.execute(
            sa.select(m.Image)
            .join(m.ImageAnnotation)
            .join(m.Building)
            .join(m.Facility)
            .join(m.CafoAnnotation)
            .group_by(m.Image.id)
            .where((m.Image.bucket != "0") & (m.Image.bucket != "1"))
            .having(
                (sa.func.count(m.CafoAnnotation.id) == 0)
                | (
                    sa.func.sum(sa.cast(m.CafoAnnotation.is_cafo, sa.Integer))
                    == sa.func.count(m.CafoAnnotation.id)
                )
            )
        )
        .unique()
        .scalars()
        .all()
    )

    facility_image_tree = STRtree(
        [ga.shape.to_shape(i.geometry) for i in facility_images]
    )
    unlabeled_image_idxs, facility_image_idxs = facility_image_tree.query(
        [ga.shape.to_shape(i.geometry) for i in unlabeled_images], predicate="touches"
    )
    facilities_with_unlabeled_adjacents = {}
    for uii, fii in zip(unlabeled_image_idxs, facility_image_idxs):
        unlabeled_image = unlabeled_images[uii]
        facility_image = facility_images[fii]
        if facility_image.id not in facilities_with_unlabeled_adjacents:
            facilities_with_unlabeled_adjacents[facility_image.id] = set()
        facilities_with_unlabeled_adjacents[facility_image.id].add(unlabeled_image)

    facility_map = {f.id: f for f in facility_images}
    for facility_id, unlabeled_images in facilities_with_unlabeled_adjacents.items():
        if verbose:
            facility_geometry = ga.shape.to_shape(facility_map[facility_id].geometry)
            facility_location = (
                facility_geometry.centroid.y,
                facility_geometry.centroid.x,
            )
            unlabeled_locations = [
                (
                    ga.shape.to_shape(ui.geometry).centroid.y,
                    ga.shape.to_shape(ui.geometry).centroid.x,
                )
                for ui in unlabeled_images
            ]
            rich.print(
                f"[yellow]Facility {facility_id} {facility_location} has unlabeled adjacent images at: {unlabeled_locations}[/yellow]"
            )
    return len(facilities_with_unlabeled_adjacents)


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
        if isinstance(expected, int):
            expected_int = expected
            expected = lambda x: x == expected  # noqa E731
            text = f"value == {expected_int}"
        else:
            text = inspect.getsource(expected)
            text = text.split("\n")[0].split(":")[1].strip("() ")
        if expected is not None and not expected(result):
            rich.print(f"[[red]Failure[/red]] {name}: expected {text} but got {result}")
        else:
            rich.print(f"[[green]OK[/green]] {name}: {result}")
