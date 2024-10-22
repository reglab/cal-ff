import geoalchemy2 as ga
import rich
import rich_click as click
import rl.utils.io
import shapely as shp
import sqlalchemy as sa

import cacafo.db.sa_models as m
from cacafo.db.session import get_sqlalchemy_session
from cacafo.transform import to_meters

CONSTANT_METHODS = []


def constant_method(func):
    CONSTANT_METHODS.append(func)
    return func


_FACILITIES = None


def cafos(session):
    global _FACILITIES
    if _FACILITIES is None:
        _FACILITIES = (
            session.execute(
                sa.select(m.Facility)
                .options(
                    sa.orm.joinedload(m.Facility.all_cafo_annotations),
                )
                .options(
                    sa.orm.joinedload(m.Facility.all_animal_type_annotations),
                )
                .options(
                    sa.orm.joinedload(m.Facility.best_permits, innerjoin=False),
                )
                .where(m.Facility.archived_at.is_(None))
            )
            .unique()
            .scalars()
            .all()
        )
    return [f for f in _FACILITIES if f.is_cafo]


@constant_method
def num_cafo_buildings(session):
    c = cafos(session)
    n_buildings = session.execute(
        sa.select(sa.func.count(m.Building.id)).where(
            m.Building.facility_id.in_([f.id for f in c])
        )
    ).scalar()
    return "{:,}".format(n_buildings)


@constant_method
def num_facilities(session):
    c = cafos(session)
    return "{:,}".format(len(c))


@constant_method
def facilities_with_no_close_permit(verbose=False):
    session = get_sqlalchemy_session()
    # get permits more than 1km from any cafo
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()
    facilities = cafos(session)
    facilities_tree = shp.STRtree(
        [to_meters(ga.shape.to_shape(f.geometry)) for f in facilities]
    )
    all_facility_ids = set(f.id for f in facilities)
    matched_facility_ids = set()
    for permit in permits:
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
        for r in registered_close_facilities:
            matched_facility_ids.add(facilities[r].id)
        for g in geocoded_close_facilities:
            matched_facility_ids.add(facilities[g].id)
    unmatched_facility_ids = all_facility_ids - matched_facility_ids
    no_close_matches = len(unmatched_facility_ids)
    return "{:,}".format(no_close_matches)


@constant_method
def facilities_with_no_close_registered_permit(verbose=False):
    session = get_sqlalchemy_session()
    # get permits more than 1km from any cafo
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()
    facilities = cafos(session)
    facilities_tree = shp.STRtree(
        [to_meters(ga.shape.to_shape(f.geometry)) for f in facilities]
    )
    matched_facility_ids = set()
    for permit in permits:
        registered = permit.registered_location and to_meters(
            ga.shape.to_shape(permit.registered_location)
        )
        registered_close_facilities = facilities_tree.query(
            registered, predicate="dwithin", distance=1000
        )
        for r in registered_close_facilities:
            matched_facility_ids.add(facilities[r].id)
    no_close_matches = len(facilities) - len(matched_facility_ids)
    return "{:,}".format(no_close_matches)


@constant_method
def facilities_with_no_best_permit(verbose=False):
    session = get_sqlalchemy_session()
    facilities = cafos(session)
    no_best_permits = sum(1 for f in facilities if not f.best_permits)
    return "{:,}".format(no_best_permits)


@constant_method
def population_estimate(session):
    return "{:,}".format(len(cafos(session)))


@constant_method
def num_initially_labeled_images(session):
    # fully labeled iamges are
    # 1. ex ante permit images
    # 2. images with bucket > 1
    # 3. adjacents
    detection_and_permit_images = (
        session.execute(
            sa.select(m.Image)
            .where(
                (
                    m.Image.bucket.is_not(None)
                    & (m.Image.bucket != "0")
                    & (m.Image.bucket != "1")
                )
            )
            .options(sa.orm.joinedload(m.Image.annotations))
            .options(
                sa.orm.joinedload(m.Image.annotations, m.ImageAnnotation.buildings)
            )
            .options(
                sa.orm.joinedload(
                    m.Image.annotations,
                    m.ImageAnnotation.buildings,
                    m.Building.facility,
                )
            )
        )
        .unique()
        .scalars()
        .all()
    )

    num_ex_ante_images = len(detection_and_permit_images)

    labeled_adjacents = set()
    for d in detection_and_permit_images:
        if d.label_status == "initially labeled" and d.is_positive:
            adjacents = d.get_adjacents(
                lazy=False, options=[sa.orm.joinedload(m.Image.annotations)]
            )
            for a in adjacents:
                if a.label_status == "unlabeled":
                    raise ValueError(f"Adjacent image {a.id} is unlabeled")
                if a.label_status == "labeled":
                    labeled_adjacents.add(a.id)
    initially_labeled_images = num_ex_ante_images + len(labeled_adjacents)
    return "{:,}".format(initially_labeled_images)


@click.command("constants")
def _cli():
    """Write all variables to file."""
    with open(rl.utils.io.get_data_path("paper", "constants.tex"), "w") as f:
        session = get_sqlalchemy_session()
        for func in CONSTANT_METHODS:
            f.write(
                r"\newcommand{{\{}}}{{{}}}".format(
                    func.__name__.replace("_", ""), func(session)
                )
            )
            f.write("\n")
            rich.print(f"Set {func.__name__.replace('_', '')} to {func(session)}")
