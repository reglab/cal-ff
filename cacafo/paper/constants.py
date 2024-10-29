import geoalchemy2 as ga
import rich
import rich_click as click
import rl.utils.io
import shapely as shp
import sqlalchemy as sa

import cacafo.db.sa_models as m
import cacafo.query
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
        subquery = cacafo.query.cafos().subquery()
        _FACILITIES = (
            session.execute(
                sa.select(m.Facility)
                .join(subquery, m.Facility.id == subquery.c.id)
                .options(
                    sa.orm.joinedload(m.Facility.all_cafo_annotations),
                )
                .options(
                    sa.orm.joinedload(m.Facility.all_animal_type_annotations),
                )
                .options(
                    sa.orm.joinedload(m.Facility.best_permits, innerjoin=False),
                )
            )
            .unique()
            .scalars()
            .all()
        )
    return _FACILITIES


@constant_method
def num_cafo_buildings(session):
    subquery = cacafo.query.cafos().subquery()
    n_buildings = (
        session.execute(
            sa.select(sa.func.count(m.Building.id))
            .select_from(m.Building)
            .join(subquery, m.Building.facility_id == subquery.c.id)
        )
        .scalars()
        .one()
        or 0
    )
    return "{:,}".format(n_buildings)


@constant_method
def num_facilities(session):
    return "{:,}".format(len(cafos(session)))


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


def facilities_with_best_permit(verbose=False):
    session = get_sqlalchemy_session()
    facilities = cafos(session)
    best_permits = sum(1 for f in facilities if f.best_permits)
    return "{:,}".format(best_permits)


@constant_method
def facilities_with_permit_animal_type(verbose=False):
    session = get_sqlalchemy_session()
    facilities = cafos(session)
    only_cow_permits = [f for f in facilities if f.animal_type_source == "permit"]
    return "{:,}".format(len(only_cow_permits))


@constant_method
def facilities_with_annotated_animal_type(verbose=False):
    session = get_sqlalchemy_session()
    facilities = cafos(session)
    only_cow_permits = [f for f in facilities if f.animal_type_source == "annotation"]
    return "{:,}".format(len(only_cow_permits))


@constant_method
def population_estimate(session):
    return "{:,}".format(len(cafos(session)))


@constant_method
def num_initially_labeled_images(session):
    return "{:,}".format(
        len(
            session.execute(cacafo.query.initially_labeled_images())
            .unique()
            .scalars()
            .all()
        )
    )


@constant_method
def permits_with_no_close_facilities(session):
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()
    facilities = cafos(session)
    facilities_tree = shp.STRtree(
        [to_meters(ga.shape.to_shape(f.geometry)) for f in facilities]
    )
    unmatched_permit_ids = set()
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
        if not len(registered_close_facilities) and not len(geocoded_close_facilities):
            unmatched_permit_ids.add(permit.id)
    no_close_matches = len(unmatched_permit_ids)
    return "{:,}".format(no_close_matches)


@constant_method
def large_active_permits_with_no_close_facilities(session):
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()
    facilities = cafos(session)
    facilities_tree = shp.STRtree(
        [to_meters(ga.shape.to_shape(f.geometry)) for f in facilities]
    )
    unmatched_permits = {}
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
        if not len(registered_close_facilities) and not len(geocoded_close_facilities):
            unmatched_permits[permit.id] = permit
    unmatched_permits = [
        u
        for u in unmatched_permits.values()
        if u.animal_count and u.animal_count >= 200 and not u.data["Termination Date"]
    ]
    return "{:,}".format(len(unmatched_permits))


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
