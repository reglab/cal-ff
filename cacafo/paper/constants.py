import geoalchemy2 as ga
import rich
import rich_click as click
import rl.utils.io
import shapely as shp
import sqlalchemy as sa

import cacafo.db.models as m
import cacafo.query
import cacafo.stats
import cacafo.stats.irr
import cacafo.stats.population
from cacafo.db.session import new_session
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


_POP_EST = None


def population_est():
    global _POP_EST
    if _POP_EST is None:
        _POP_EST = cacafo.stats.population.estimate_population()
    return _POP_EST


_IMAGE_SURVEY = None


def img_survey():
    global _IMAGE_SURVEY
    if _IMAGE_SURVEY is None:
        _IMAGE_SURVEY = cacafo.stats.population.Survey.from_db()
    return _IMAGE_SURVEY


_IMAGES = None


def images(session):
    global _IMAGES
    if _IMAGES is None:
        _IMAGES = (
            session.execute(
                (sa.select(m.Image)).options(
                    sa.orm.selectinload(m.Image.annotations),
                    sa.orm.selectinload(m.Image.county),
                )
            )
            .unique()
            .all()
        )
    return _IMAGES


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
    session = new_session()
    no_close_matches = (
        session.execute(cacafo.query.unpermitted_facilities().select())
        .unique()
        .scalars()
        .all()
    )
    return "{:,}".format(len(no_close_matches))


@constant_method
def facilities_with_no_close_registered_permit(verbose=False):
    session = new_session()
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
    session = new_session()
    facilities = cafos(session)
    no_best_permits = sum(1 for f in facilities if not f.best_permits)
    return "{:,}".format(no_best_permits)


@constant_method
def facilities_with_best_permit(verbose=False):
    session = new_session()
    facilities = cafos(session)
    best_permits = sum(1 for f in facilities if f.best_permits)
    return "{:,}".format(best_permits)


@constant_method
def facilities_with_permit_animal_type(verbose=False):
    session = new_session()
    facilities = cafos(session)
    only_cow_permits = [f for f in facilities if f.animal_type_source == "permit"]
    return "{:,}".format(len(only_cow_permits))


@constant_method
def facilities_with_annotated_animal_type(verbose=False):
    session = new_session()
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


@constant_method
def total_permits(session):
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()
    return "{:,}".format(len(permits))


@constant_method
def large_permits(session):
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()

    large_permits = [
        p for p in permits if (p.animal_count is not None) and p.animal_count >= 200
    ]
    return "{:,}".format(len(large_permits))


@constant_method
def small_permits(session):
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()

    small_permits = [
        p for p in permits if (p.animal_count is not None) and p.animal_count < 200
    ]
    return "{:,}".format(len(small_permits))


@constant_method
def no_animal_count_permits(session):
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()

    no_animal_count_permits = [p for p in permits if p.animal_count is None]
    return "{:,}".format(len(no_animal_count_permits))


@constant_method
def fac_to_im_ratio(session):
    return "{:.5f}".format(
        session.execute(cacafo.stats.population.number_of_images_per_facility())
        .scalars()
        .one()
    )


@constant_method
def completeness_est(session):
    survey = img_survey()
    # this est is images population, not number of facilities
    pop_est = survey.population().point
    unseen_images_est = pop_est - survey.positive()
    unseen_facilities_est = unseen_images_est / float(fac_to_im_ratio(session))
    observed_facilities = len(cafos(session))
    return r"{:.2f}\%".format(
        100 * observed_facilities / (observed_facilities + unseen_facilities_est)
    )


@constant_method
def completeness_lower(session):
    survey = img_survey()
    # this est is images population, not number of facilities
    pop_est = survey.population().upper
    unseen_images_est = pop_est - survey.positive()
    unseen_facilities_est = unseen_images_est / float(fac_to_im_ratio(session))
    observed_facilities = len(cafos(session))
    return r"{:.2f}\%".format(
        100 * observed_facilities / (observed_facilities + unseen_facilities_est)
    )


@constant_method
def FNR_est(session):
    survey = img_survey()
    survey_0 = cacafo.stats.population.Survey(
        strata=[stratum for stratum in survey.strata if "0:" in stratum.name],
        post_hoc_positive=0,
    )
    population_0 = cacafo.stats.population.stratum_f_estimator(survey_0)

    total_images = sum([stratum.total for stratum in survey_0.strata])
    FN_est = population_0.point
    return "{:.4f}".format(FN_est / total_images)


@constant_method
def FNR_upper(session):
    survey = img_survey()
    survey_0 = cacafo.stats.population.Survey(
        strata=[stratum for stratum in survey.strata if "0:" in stratum.name],
        post_hoc_positive=0,
    )
    population_0 = cacafo.stats.population.stratum_f_estimator(survey_0)

    total_images = sum([stratum.total for stratum in survey_0.strata])
    FN_upper = population_0.upper
    return "{:.4f}".format(FN_upper / total_images)


@constant_method
def unobserved_FN_upper(session):
    survey = img_survey()
    survey_0 = cacafo.stats.population.Survey(
        strata=[stratum for stratum in survey.strata if "0:" in stratum.name],
        post_hoc_positive=0,
    )
    population_0 = cacafo.stats.population.stratum_f_estimator(survey_0)
    FN_upper = population_0.upper
    observed_0 = sum([stratum.positive for stratum in survey_0.strata])

    return "{:,}".format(FN_upper - observed_0)


@constant_method
def unlabeled_negative_count(session):
    survey = img_survey()
    survey_0 = cacafo.stats.population.Survey(
        strata=[stratum for stratum in survey.strata if "0:" in stratum.name],
        post_hoc_positive=0,
    )
    unlabeled_images = sum([stratum.unlabeled for stratum in survey_0.strata])
    return "{:,}".format(unlabeled_images)


@constant_method
def unobserved_TP_est(session):
    survey = img_survey()
    survey_1 = cacafo.stats.population.Survey(
        strata=[stratum for stratum in survey.strata if "1:" in stratum.name],
        post_hoc_positive=0,
    )
    population_1 = cacafo.stats.population.stratum_f_estimator(survey_1)
    observed_1 = sum([stratum.positive for stratum in survey_1.strata])

    TP_est = population_1.point
    return "{:,}".format(TP_est - observed_1)


@constant_method
def unobserved_TP_upper(session):
    survey = img_survey()
    survey_1 = cacafo.stats.population.Survey(
        strata=[stratum for stratum in survey.strata if "1:" in stratum.name],
        post_hoc_positive=0,
    )
    population_1 = cacafo.stats.population.stratum_f_estimator(survey_1)
    observed_1 = sum([stratum.positive for stratum in survey_1.strata])

    TP_upper = population_1.upper
    return "{:,}".format(TP_upper - observed_1)


@constant_method
def positive_tiles_est(session):
    pop_est = population_est().point

    return "{:,}".format(pop_est)


@constant_method
def positive_tiles_upper(session):
    pop_upper = population_est().upper

    return "{:,}".format(pop_upper)


@constant_method
def total_labeled(session):
    survey = img_survey()
    labeled = sum([stratum.labeled for stratum in survey.strata])

    return "{:,}".format(labeled)


@constant_method
def pct_ca_labeled(session):
    survey = img_survey()
    labeled = sum([stratum.labeled for stratum in survey.strata])
    area_of_CA = 481000
    return r"{:.3f}\%".format(100 * labeled / area_of_CA)


@constant_method
def total_buildings(session):
    n_buildings = (
        session.execute(sa.select(sa.func.count(m.Building.id)).select_from(m.Building))
        .scalars()
        .one()
        or 0
    )
    return "{:,}".format(n_buildings)


@constant_method
def total_facilities(session):
    n_facilities = (
        session.execute(
            sa.select(sa.func.count(m.Facility.id))
            .select_from(m.Facility)
            .where(m.Facility.archived_at.is_(None))
        )
        .scalars()
        .one()
        or 0
    )
    return "{:,}".format(n_facilities)


@constant_method
def pct_image_labeled(session):
    subquery = cacafo.query.labeled_images().subquery()
    labeled = (
        session.execute(sa.select(sa.func.count(subquery.c.id)).select_from(subquery))
        .scalars()
        .one()
    )
    total = (
        session.execute(
            sa.select(sa.func.count(m.Image.id)).where(m.Image.bucket.is_not(None))
        )
        .scalars()
        .one()
    )
    return r"{:.2f}\%".format(100 * labeled / total)


@constant_method
def irr(session):
    return "{:.2f}".format(cacafo.stats.irr.label_balanced_cohens_kappa(session))


@constant_method
def agreement_pct_dating(session):
    pass


@constant_method
def removed_buildings_dating(session):
    pass


@constant_method
def removed_facilities_dating(session):
    n_facilities = (
        session.execute(
            sa.select(sa.func.count(m.Facility.id))
            .select_from(m.Facility)
            .join(m.ConstructionAnnotation)
            .where(m.Facility)
        )
        .scalars()
        .one()
        or 0
    )
    return "{:,}".format(n_facilities)


@constant_method
def removed_facilities_typing(session):
    pass


@constant_method
def removed_pct_typing(session):
    pass


"""
\newcommand{\agreementpctdating}{86\%} %pct of labeled buildings not removed in dating procedure
\newcommand{\removedbuildingsdating}{4,648}
\newcommand{\partiallyremovedfacilitiesdating}{677}
\newcommand{\removedfacilitiesdating}{597}
\newcommand{\removedfacilitiestyping}{47}
\newcommand{\removedpcttyping}{2\%}
"""


@click.command("constants", help="Write all paper constants to file.")
def _cli():
    """Write all variables to file."""
    with open(rl.utils.io.get_data_path("paper", "constants.tex"), "w") as f:
        session = new_session()
        for func in CONSTANT_METHODS:
            f.write(
                r"\newcommand{{\{}}}{{{}}}".format(
                    func.__name__.replace("_", ""), func(session)
                )
            )
            f.write("\n")
            rich.print(f"Set {func.__name__.replace('_', '')} to {func(session)}")
