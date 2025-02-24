import csv
import math

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
def facilities_with_no_close_permit(session):
    no_close_matches = (
        session.execute(cacafo.query.unpermitted_cafos().select())
        .unique()
        .scalars()
        .all()
    )
    return "{:,}".format(len(no_close_matches))


@constant_method
def percent_more_facilities(session):
    no_close_matches = (
        session.execute(cacafo.query.unpermitted_cafos().select())
        .unique()
        .scalars()
        .all()
    )
    permits = session.execute(sa.select(m.Permit)).unique().scalars().all()

    return r"{:.1f}\%".format((len(no_close_matches) / len(permits)) * 100)


@constant_method
def facilities_with_no_close_registered_permit(session):
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
def percent_facilities_with_no_close_registered_permit(session):
    facilities = cafos(session)
    unpermitted = int(
        facilities_with_no_close_registered_permit(session).replace(",", "")
    )
    return r"{:.1f}\%".format((unpermitted / len(facilities)) * 100)


@constant_method
def facilities_with_no_best_permit(session):
    facilities = cafos(session)
    no_best_permits = sum(1 for f in facilities if not f.best_permits)
    return "{:,}".format(no_best_permits)


@constant_method
def percent_facilities_with_no_best_permit(session):
    facilities = cafos(session)
    no_best_permits = sum(1 for f in facilities if not f.best_permits)
    return r"{:.1f}\%".format((no_best_permits / len(facilities)) * 100)


@constant_method
def facilities_with_best_permit(session):
    facilities = cafos(session)
    best_permits = sum(1 for f in facilities if f.best_permits)
    return "{:,}".format(best_permits)


@constant_method
def percent_facilities_with_best_permit(session):
    facilities = cafos(session)
    best_permits = sum(1 for f in facilities if f.best_permits)
    return r"{:.1f}\%".format((best_permits / len(facilities)) * 100)


@constant_method
def facilities_with_permit_animal_type(session):
    facilities = cafos(session)
    only_cow_permits = [f for f in facilities if f.animal_type_source == "permit"]
    return "{:,}".format(len(only_cow_permits))


@constant_method
def facilities_with_annotated_animal_type(session):
    facilities = cafos(session)
    only_cow_permits = [f for f in facilities if f.animal_type_source == "annotation"]
    return "{:,}".format(len(only_cow_permits))


@constant_method
def population_estimate(session):
    img_est = population_est()
    ftir = (
        session.execute(cacafo.stats.population.number_of_images_per_facility())
        .scalars()
        .one()
    )
    return "{:,}".format(round((img_est / ftir).point))


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


# @constant_method
# def permits_with_no_close_facilities(session):
#     permits = session.execute(sa.select(m.Permit)).unique().scalars().all()
#     facilities = cafos(session)
#     facilities_tree = shp.STRtree(
#         [to_meters(ga.shape.to_shape(f.geometry)) for f in facilities]
#     )
#     unmatched_permit_ids = set()
#     for permit in permits:
#         registered = permit.registered_location and to_meters(
#             ga.shape.to_shape(permit.registered_location)
#         )
#         geocoded = permit.geocoded_address_location and to_meters(
#             ga.shape.to_shape(permit.geocoded_address_location)
#         )
#         registered_close_facilities = facilities_tree.query(
#             registered, predicate="dwithin", distance=1000
#         )
#         geocoded_close_facilities = facilities_tree.query(
#             geocoded, predicate="dwithin", distance=1000
#         )
#         if not len(registered_close_facilities) and not len(geocoded_close_facilities):
#             unmatched_permit_ids.add(permit.id)
#     no_close_matches = len(unmatched_permit_ids)
#     return "{:,}".format(no_close_matches)


@constant_method
def permits_with_no_close_facilities(session):
    no_close_matches = (
        session.execute(cacafo.query.permits_without_cafo().select())
        .unique()
        .scalars()
        .all()
    )
    return "{:,}".format(len(no_close_matches))


@constant_method
def permits_with_no_best_facility(session):
    return "{:,}".format(
        len(
            session.execute(sa.select(m.Permit).where(m.Permit.facility_id.is_(None)))
            .scalars()
            .all()
        )
    )


@constant_method
def permits_with_best_facility(session):
    return "{:,}".format(
        len(
            session.execute(sa.select(m.Permit).where(m.Permit.facility_id.isnot(None)))
            .scalars()
            .all()
        )
    )


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
    return "{:.2f}".format(
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
    return r"{:}\%".format(
        round(100 * observed_facilities / (observed_facilities + unseen_facilities_est))
    )


@constant_method
def completeness_lower(session):
    survey = img_survey()
    # this est is images population, not number of facilities
    pop_est = survey.population().upper
    unseen_images_est = pop_est - survey.positive()
    unseen_facilities_est = unseen_images_est / float(fac_to_im_ratio(session))
    observed_facilities = len(cafos(session))
    return r"{:}\%".format(
        round(100 * observed_facilities / (observed_facilities + unseen_facilities_est))
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
        session.execute(
            sa.select(sa.func.count(m.Image.id)).where(
                m.Image.id.in_(sa.select(subquery.c.id))
            )
        )
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
def pct_image_initially_labeled(session):
    subquery = cacafo.query.initially_labeled_images().subquery()
    labeled = (
        session.execute(
            sa.select(sa.func.count(m.Image.id)).where(
                m.Image.id.in_(sa.select(subquery.c.id))
            )
        )
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
def irr_positive_images(session):
    subq = cacafo.query.positive_images().subquery()
    return "{}".format(
        session.execute(
            sa.select(sa.func.count(subq.c.id.distinct())).select_from(subq)
        )
        .scalars()
        .one()
    )


@constant_method
def irr_labeled_negative_images(session):
    subq = cacafo.query.labeled_negative_images().subquery()
    return "{}".format(
        session.execute(
            sa.select(sa.func.count(subq.c.id.distinct())).select_from(subq)
        )
        .scalars()
        .one()
    )


@constant_method
def irr_high_confidence_negative_images(session):
    subq = cacafo.query.high_confidence_negative_images().subquery()
    return "{}".format(
        session.execute(
            sa.select(sa.func.count(subq.c.id.distinct())).select_from(subq)
        )
        .scalars()
        .one()
    )


@constant_method
def irr_low_confidence_negative_images(session):
    subq = cacafo.query.low_confidence_negative_images().subquery()
    return "{}".format(
        session.execute(
            sa.select(sa.func.count(subq.c.id.distinct())).select_from(subq)
        )
        .scalars()
        .one()
    )


@constant_method
def removed_pct_dating(session):
    n_removed = int(removed_facilities_dating(session).replace(",", ""))
    n_total = int(total_facilities(session).replace(",", ""))
    return "{:.2f}\\%".format(100 * n_removed / n_total)


@constant_method
def removed_facilities_dating(session):
    facilities = session.execute(
        sa.select(
            m.Facility,
            # whether there is a construction dating annotation
            sa.func.sum(
                sa.cast(
                    sa.case(
                        (
                            m.CafoAnnotation.annotation_phase == "construction dating",
                            1,
                        ),
                        else_=0,
                    ),
                    sa.Integer,
                )
            ).label("construction_dating_no_cafo"),
            sa.func.sum(
                sa.cast(
                    sa.case(
                        (
                            m.CafoAnnotation.annotation_phase == "animal typing",
                            1,
                        ),
                        else_=0,
                    ),
                    sa.Integer,
                )
            ).label("animal_typing_no_cafo"),
        )
        .join(m.CafoAnnotation)
        .where(m.Facility.archived_at.is_(None) & m.CafoAnnotation.is_cafo.is_(False))
        .group_by(m.Facility.id)
    )
    n_facilities_removed = len([f for f in facilities if f[1]])
    return "{:,}".format(n_facilities_removed)


def removed_facilities_both(session):
    facilities_removed_either = (
        sa.select(m.Facility.id, m.CafoAnnotation.annotation_phase)
        .select_from(m.Facility)
        .join(m.CafoAnnotation, isouter=True)
        .group_by(m.Facility.id, m.CafoAnnotation.annotation_phase)
        .having(
            (
                sa.func.sum(sa.cast(m.CafoAnnotation.is_cafo, sa.Integer))
                < sa.func.count(m.CafoAnnotation.id)
            )
            & (sa.func.count(m.CafoAnnotation.id) != 0)
        )
        .where(m.Facility.archived_at.is_(None))
        .subquery()
    )
    facilities_removed = (
        sa.select(facilities_removed_either.c.id)
        .select_from(facilities_removed_either)
        .group_by(facilities_removed_either.c.id)
        .having(
            sa.func.count(facilities_removed_either.c.annotation_phase.distinct()) == 2
        )
        .subquery()
    )
    n_facilities_removed = (
        session.execute(
            sa.select(sa.func.count(facilities_removed.c.id.distinct())).select_from(
                facilities_removed
            )
        )
        .scalars()
        .one()
        or 0
    )
    return "{:,}".format(n_facilities_removed)


@constant_method
def removed_facilities_typing(session):
    facilities = session.execute(
        sa.select(
            m.Facility,
            # whether there is a construction dating annotation
            sa.func.sum(
                sa.cast(
                    sa.case(
                        (
                            m.CafoAnnotation.annotation_phase == "construction dating",
                            1,
                        ),
                        else_=0,
                    ),
                    sa.Integer,
                )
            ).label("construction_dating_no_cafo"),
            sa.func.sum(
                sa.cast(
                    sa.case(
                        (
                            m.CafoAnnotation.annotation_phase == "animal typing",
                            1,
                        ),
                        else_=0,
                    ),
                    sa.Integer,
                )
            ).label("animal_typing_no_cafo"),
        )
        .join(m.CafoAnnotation)
        .where(m.Facility.archived_at.is_(None) & m.CafoAnnotation.is_cafo.is_(False))
        .group_by(m.Facility.id)
    )
    n_facilities_removed = len([f for f in facilities if f[2] and not f[1]])
    return "{:,}".format(n_facilities_removed)


@constant_method
def removed_pct_typing(session):
    n_removed = int(removed_facilities_typing(session).replace(",", ""))
    n_total = int(total_facilities(session).replace(",", ""))
    return "{:.2f}\\%".format(100 * n_removed / n_total)


_FACILITIES_WITHIN_URBAN_MASK = None


def _facilities_within_urban_mask(session):
    global _FACILITIES_WITHIN_URBAN_MASK
    if _FACILITIES_WITHIN_URBAN_MASK is None:
        urban_mask = session.execute(sa.select(m.UrbanMask)).scalars().all()
        subquery = cacafo.query.positive_images()
        positive_images = (
            session.execute(
                sa.select(m.Image).where(m.Image.id.in_(sa.select(subquery.c.id)))
            )
            .unique()
            .scalars()
            .all()
        )
        urban_mask_geoms = [ga.shape.to_shape(um.geometry) for um in urban_mask]
        urban_mask_tree = shp.STRtree(urban_mask_geoms)
        positive_image_geoms = [
            ga.shape.to_shape(ui.geometry) for ui in positive_images
        ]
        positive_image_idxs, urban_mask_idxs = urban_mask_tree.query(
            positive_image_geoms, predicate="intersects"
        )
        images_intersecting_with_urban_mask = {
            positive_images[pii]: urban_mask[umi]
            for pii, umi in zip(positive_image_idxs, urban_mask_idxs)
        }

        union_urban_mask_geom = shp.ops.unary_union(urban_mask_geoms)
        images_within_urban_mask = []
        for image, mask in images_intersecting_with_urban_mask.items():
            if (
                image.shp_geometry.area * 0.7
                < shp.intersection(image.shp_geometry, union_urban_mask_geom).area
            ):
                images_within_urban_mask.append(image.id)
        # get facilities on these images
        subquery = cacafo.query.cafos()
        facilities = (
            session.execute(
                sa.select(m.Facility)
                .join(m.Building)
                .join(m.ImageAnnotation)
                .where(m.Facility.id.in_(sa.select(subquery.c.id)))
                .group_by(m.Facility.id)
                # having all images in the urban mask
                .having(
                    sa.func.count(m.ImageAnnotation.id)
                    == sa.func.sum(
                        sa.case(
                            (
                                m.ImageAnnotation.image_id.in_(
                                    images_within_urban_mask
                                ),
                                1,
                            ),
                            else_=0,
                        )
                    )
                )
            )
            .unique()
            .scalars()
            .all()
        )
        _FACILITIES_WITHIN_URBAN_MASK = facilities
    return _FACILITIES_WITHIN_URBAN_MASK


@constant_method
def num_facilities_within_urban_mask(session):
    facilities = _facilities_within_urban_mask(session)
    return "{:,}".format(len(facilities))


@constant_method
def pct_facilities_within_urban_mask(session):
    facilities = _facilities_within_urban_mask(session)
    return "{:.2f}\\%".format(100 * len(facilities) / len(cafos(session)))


@constant_method
def num_urban_masks(session):
    session = new_session()
    urban_mask = session.execute(sa.select(m.UrbanMask)).scalars().all()
    return "{:,}".format(len(urban_mask))


_UNAPPLIED_URBAN_MASKS = None


def _unapplied_urban_masks(session):
    global _UNAPPLIED_URBAN_MASKS
    if _UNAPPLIED_URBAN_MASKS is None:
        session = new_session()
        urban_masks = session.execute(sa.select(m.UrbanMask)).scalars().all()
        image_csv = cacafo.data.source.get("images.csv")
        with open(image_csv, "r") as f:
            image_geometries = [
                shp.box(
                    float(row["lon_min"]),
                    float(row["lat_min"]),
                    float(row["lon_max"]),
                    float(row["lat_max"]),
                )
                for row in csv.DictReader(f)
                if row["bucket"]
            ]
        urban_mask_geoms = [ga.shape.to_shape(um.geometry) for um in urban_masks]
        urban_mask_tree = shp.STRtree(urban_mask_geoms)
        image_idxs, urban_mask_idxs = urban_mask_tree.query(
            image_geometries, predicate="within"
        )
        urban_masks_with_images = {urban_masks[ui] for ui in urban_mask_idxs}
        _UNAPPLIED_URBAN_MASKS = urban_masks_with_images
    return _UNAPPLIED_URBAN_MASKS


@constant_method
def urban_masks_originally_not_applied(session):
    return len(_unapplied_urban_masks(session))


@constant_method
def urban_mask_pct_of_state(session):
    session = new_session()
    urban_mask = session.execute(sa.select(m.UrbanMask)).scalars().all()
    total_area = sum([um.shp_geometry_meters.area for um in urban_mask]) // 1_000_000
    state_area = 423970
    return "{:.2f}\\%".format(100 * total_area / state_area)


@constant_method
def urban_masks_originally_not_applied_pct_of_state(session):
    urban_masks_with_images = _unapplied_urban_masks(session)
    total_area = (
        sum([um.shp_geometry_meters.area for um in urban_masks_with_images])
        // 1_000_000
    )
    state_area = 423970
    return "{:.2f}\\%".format(100 * total_area / state_area)


@constant_method
def expected_facilities_in_remaining_urban_mask(session):
    facilities_in_labeled_urban_mask = _facilities_within_urban_mask(session)
    labeled_mask_area = sum(
        [um.shp_geometry_meters.area for um in _unapplied_urban_masks(session)]
    )
    total_mask_area = sum(
        [
            um.shp_geometry_meters.area
            for um in session.execute(sa.select(m.UrbanMask)).scalars().all()
        ]
    )
    expected_facilities = len(facilities_in_labeled_urban_mask) * (
        1 - (labeled_mask_area / total_mask_area)
    )
    # round up to nearest int
    return "{:,}".format(math.ceil(expected_facilities))


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
