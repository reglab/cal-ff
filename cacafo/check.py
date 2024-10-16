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

tbd_images = [
    "Amador_5_8192_27648",
    "Del Norte_0_28672_8192",
    "Fresno_1_32768_26624",
    "Fresno_3_13312_23552",
    "Fresno_3_28672_23552",
    "Fresno_9_0_28672",
    "Fresno_9_7168_27648",
    "Fresno_9_7168_29696",
    "Fresno_9_23552_34816",
    "Fresno_19_23552_10240",
    "Fresno_19_26624_13312",
    "Fresno_22_33792_17408",
    "Fresno_32_3072_21504",
    "Fresno_32_20480_23552",
    "Fresno_32_22528_24576",
    "Fresno_32_25600_31744",
    "Fresno_34_1024_12288",
    "Fresno_34_2048_18432",
    "Fresno_34_4096_4096",
    "Fresno_34_6144_8192",
    "Fresno_34_8192_12288",
    "Fresno_34_14336_10240",
    "Fresno_34_23552_7168",
    "Glenn_3_16384_26624",
    "Humboldt_9_34816_23552",
    "Imperial_1_0_27648",
    "Imperial_1_3072_12288",
    "Imperial_7_26624_2048",
    "Kern_1_9216_30720",
    "Kern_1_18432_19456",
    "Kern_2_7168_31744",
    "Kern_8_16384_3072",
    "Kern_22_8192_11264",
    "Kings_6_18432_5120",
    "Kings_6_18432_17408",
    "Kings_6_19456_33792",
    "Kings_6_30720_29696",
    "Kings_7_7168_6144",
    "Kings_7_7168_14336",
    "Kings_7_8192_31744",
    "Kings_7_13312_9216",
    "Madera_1_8192_2048",
    "Madera_1_15360_10240",
    "Madera_6_10240_5120",
    "Madera_6_30720_24576",
    "Madera_8_24576_1024",
    "Madera_12_33792_6144",
    "Marin_2_22528_19456",
    "Marin_2_25600_16384",
    "Marin_2_29696_5120",
    "Mendocino_7_32768_23552",
    "Merced_0_5120_32768",
    "Merced_0_14336_12288",
    "Merced_1_4096_6144",
    "Merced_1_11264_10240",
    "Merced_4_0_21504",
    "Merced_4_1024_27648",
    "Merced_4_24576_0",
    "Merced_4_28672_0",
    "Merced_8_14336_36864",
    "Merced_10_32768_32768",
    "Sacramento_3_15360_4096",
    "Sacramento_3_23552_2048",
    "Sacramento_3_25600_1024",
    "Sacramento_7_1024_9216",
    "Diego_8_33792_19456",
    "San Joaquin_0_12288_21504",
    "San Joaquin_2_1024_25600",
    "San Joaquin_2_26624_3072",
    "San Joaquin_4_17408_19456",
    "San Joaquin_4_24576_18432",
    "San Joaquin_4_36864_26624",
    "Santa Barbara_1_18432_24576",
    "Santa Clara_0_6144_9216",
    "Siskiyou_19_14336_17408",
    "Sonoma_6_24576_21504",
    "Sonoma_7_5120_19456",
    "Sonoma_7_8192_20480",
    "Sonoma_7_12288_20480",
    "Sonoma_7_17408_26624",
    "Sonoma_7_22528_16384",
    "Sonoma_7_35840_23552",
    "Stanislaus_1_0_23552",
    "Stanislaus_1_5120_29696",
    "Stanislaus_1_9216_1024",
    "Stanislaus_5_27648_22528",
    "Stanislaus_7_32768_26624",
    "Stanislaus_9_3072_0",
    "Stanislaus_11_0_16384",
    "Sutter_0_5120_3072",
    "Tulare_0_6144_14336",
    "Tulare_0_11264_20480",
    "Tulare_0_15360_32768",
    "Tulare_0_17408_7168",
    "Tulare_0_21504_35840",
    "Tulare_0_29696_12288",
    "Tulare_0_32768_35840",
    "Tulare_6_3072_20480",
    "Tulare_7_13312_31744",
    "Tulare_11_20480_28672",
    "Tulare_11_20480_30720",
    "Tulare_12_16384_6144",
    "Tulare_12_19456_3072",
    "Tulare_12_19456_9216",
    "Tulare_12_36864_0",
]


def check(expected=None):
    def wrapper(func):
        checks[func] = expected
        return func

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
            .where(
                (m.ImageAnnotation.id.is_(None))
                & (m.Image.bucket.is_not(None))
                & (m.Image.name.notin_(tbd_images))
            )
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
    labeled_images = (
        session.execute(
            sa.select(m.Image)
            .join(m.ImageAnnotation, isouter=True)
            .where(m.ImageAnnotation.id.isnot(None))
        )
        .unique()
        .scalars()
        .all()
    )

    labeled_image_tree = STRtree(
        [ga.shape.to_shape(i.geometry) for i in labeled_images]
    )
    _unlabeled_images = []
    for ui in unlabeled_images:
        intersections = labeled_image_tree.query(
            ga.shape.to_shape(ui.geometry), predicate="intersects"
        )
        if not len(intersections):
            _unlabeled_images.append(ui)
            continue
        intersection_areas = [
            ga.shape.to_shape(ui.geometry)
            .intersection(ga.shape.to_shape(labeled_images[i].geometry))
            .area
            for i in intersections
        ]
        if all([ia == 0 for ia in intersection_areas]):
            _unlabeled_images.append(ui)
    unlabeled_images = _unlabeled_images

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
    # return sum([list(v) for v in facilities_with_unlabeled_adjacents.values()], [])
    return len(facilities_with_unlabeled_adjacents)


@check(expected=0)
def unremoved_images_intersecting_with_urban_mask(verbose=False):
    session = get_sqlalchemy_session()
    urban_mask = session.execute(sa.select(m.UrbanMask)).scalars().all()
    unremoved_images = (
        session.execute(sa.select(m.Image).where(m.Image.bucket.is_not(None)))
        .scalars()
        .all()
    )
    urban_mask_geoms = [ga.shape.to_shape(um.geometry) for um in urban_mask]
    urban_mask_tree = STRtree(urban_mask_geoms)
    unremoved_image_geoms = [ga.shape.to_shape(ui.geometry) for ui in unremoved_images]
    unremoved_image_ids, urban_mask_ids = urban_mask_tree.query(
        unremoved_image_geoms, predicate="within"
    )
    for ui in unremoved_image_ids:
        if verbose:
            rich.print(
                f"[yellow]Image {unremoved_images[ui].id} intersects with urban mask[/yellow]"
            )
    return len(unremoved_image_ids)


@check(expected=0)
def positive_images_intersecting_with_urban_mask(verbose=False):
    session = get_sqlalchemy_session()
    urban_mask = session.execute(sa.select(m.UrbanMask)).scalars().all()
    positive_images = (
        session.execute(
            sa.select(m.Image)
            .join(m.ImageAnnotation)
            .join(m.Building)
            .where(m.Image.bucket.is_not(None))
        )
        .unique()
        .scalars()
        .all()
    )
    urban_mask_geoms = [ga.shape.to_shape(um.geometry) for um in urban_mask]
    urban_mask_tree = STRtree(urban_mask_geoms)
    positive_image_geoms = [ga.shape.to_shape(ui.geometry) for ui in positive_images]
    positive_image_ids, urban_mask_ids = urban_mask_tree.query(
        positive_image_geoms, predicate="within"
    )
    for ui in positive_image_ids:
        if verbose:
            image_location = (
                ga.shape.to_shape(positive_images[ui].geometry).centroid.y,
                ga.shape.to_shape(positive_images[ui].geometry).centroid.x,
            )
            rich.print(
                f"[yellow]Image {positive_images[ui].id} {image_location} intersects with urban mask[/yellow]"
            )
    return len(positive_image_ids)


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
