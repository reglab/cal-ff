import sqlalchemy as sa

import cacafo.db.models as m

CA_SRID = 3311


def cafos():
    return (
        sa.select(m.Facility)
        .join(m.CafoAnnotation, isouter=True)
        .join(m.ConstructionAnnotation, isouter=True)
        .group_by(m.Facility.id)
        .having(
            (
                sa.func.sum(sa.cast(m.CafoAnnotation.is_cafo, sa.Integer))
                == sa.func.count(m.CafoAnnotation.id)
            )
            | (sa.func.count(m.CafoAnnotation.id) == 0)
        )
        .having(
            sa.or_(
                sa.func.max(m.ConstructionAnnotation.destruction_upper_bound)
                > "2017-12-31",
                sa.func.max(m.ConstructionAnnotation.destruction_upper_bound).is_(None),
                sa.func.count(m.ConstructionAnnotation.id) == 0,
            )
        )
        .where(m.Facility.archived_at.is_(None))
    )


def permitted_cafos():
    cafos_subquery = cafos().subquery()
    return (
        sa.select(m.Facility)
        .where(m.Facility.id.in_(sa.select(cafos_subquery.c.id)))
        .join(
            m.Permit,
            sa.or_(
                sa.func.ST_DWithin(
                    m.Permit.registered_location,
                    m.Facility.geometry,
                    1000,
                ),
                sa.func.ST_DWithin(
                    m.Permit.geocoded_address_location,
                    m.Facility.geometry,
                    1000,
                ),
            ),
        )
        .group_by(m.Facility.id)
    )


def unpermitted_cafos():
    cafos_subquery = cafos().subquery()
    return (
        sa.select(m.Facility)
        .where(m.Facility.id.in_(sa.select(cafos_subquery.c.id)))
        .join(
            m.Permit,
            sa.or_(
                sa.func.ST_DWithin(
                    m.Permit.registered_location,
                    m.Facility.geometry,
                    1000,
                ),
                sa.func.ST_DWithin(
                    m.Permit.geocoded_address_location,
                    m.Facility.geometry,
                    1000,
                ),
            ),
            isouter=True,
        )
        .group_by(m.Facility.id)
        .having(sa.func.count(m.Permit.id) == 0)
    )


def permitted_cafos_active_only():
    cafos_subquery = cafos().subquery()
    active_permits = (
        sa.select(m.Permit)
        .where(m.Permit.data["Regulatory Measure Status"].astext == "Active")
        .subquery()
    )
    return (
        sa.select(m.Facility)
        .where(m.Facility.id.in_(sa.select(cafos_subquery.c.id)))
        .join(
            active_permits,
            sa.or_(
                sa.func.ST_DWithin(
                    active_permits.c.registered_location,
                    m.Facility.geometry,
                    1000,
                ),
                sa.func.ST_DWithin(
                    active_permits.c.geocoded_address_location,
                    m.Facility.geometry,
                    1000,
                ),
            ),
        )
        .group_by(m.Facility.id)
    )


def permitted_cafos_historical_only():
    cafos_subquery = cafos().subquery()
    active_permitted_cafos = permitted_cafos_active_only().subquery()
    historical_permits = (
        sa.select(m.Permit)
        .where(m.Permit.data["Regulatory Measure Status"].astext == "Historical")
        .subquery()
    )
    return (
        sa.select(m.Facility)
        .where(
            sa.and_(
                m.Facility.id.in_(sa.select(cafos_subquery.c.id)),
                m.Facility.id.not_in(sa.select(active_permitted_cafos.c.id)),
            )
        )
        .join(
            historical_permits,
            sa.or_(
                sa.func.ST_DWithin(
                    historical_permits.c.registered_location,
                    m.Facility.geometry,
                    1000,
                ),
                sa.func.ST_DWithin(
                    historical_permits.c.geocoded_address_location,
                    m.Facility.geometry,
                    1000,
                ),
            ),
        )
        .group_by(m.Facility.id)
    )


def permits_without_cafo():
    cafos_subquery = cafos().subquery()
    return (
        sa.select(m.Permit)
        .join(
            cafos_subquery,
            sa.or_(
                sa.func.ST_DWithin(
                    m.Permit.registered_location,
                    cafos_subquery.c.geometry,
                    1000,
                ),
                sa.func.ST_DWithin(
                    m.Permit.geocoded_address_location,
                    cafos_subquery.c.geometry,
                    1000,
                ),
            ),
            isouter=True,
        )
        .group_by(m.Permit.id)
        .having(sa.func.count(cafos_subquery.c.id) == 0)
    )


def permits_with_cafo():
    cafos_subquery = cafos().subquery()
    return (
        sa.select(m.Permit)
        .join(
            cafos_subquery,
            sa.or_(
                sa.func.ST_DWithin(
                    m.Permit.registered_location,
                    cafos_subquery.c.geometry,
                    1000,
                ),
                sa.func.ST_DWithin(
                    m.Permit.geocoded_address_location,
                    cafos_subquery.c.geometry,
                    1000,
                ),
            ),
        )
        .group_by(m.Permit.id)
        .having(sa.func.count(cafos_subquery.c.id) > 0)
    )


def positive_images():
    cafo_query = cafos().subquery()
    return (
        sa.select(m.Image)
        .join(m.ImageAnnotation)
        .join(m.Building)
        .join(m.Facility)
        .join(m.CafoAnnotation, isouter=True)
        .group_by(m.Image.id)
        .where(m.Facility.id.in_(sa.select(cafo_query.c.id)))
    )


def labeled_negative_images():
    """
    Images that have been labeled with bounding boxes but
    later determined not to have a CAFO.
    """
    positive_image_subquery = positive_images().subquery()
    return (
        sa.select(m.Image)
        .join(m.ImageAnnotation)
        .join(m.Building)
        .join(m.Facility)
        .group_by(m.Image.id)
        .where(m.Image.id.notin_(sa.select(positive_image_subquery.c.id)))
    )


def high_confidence_negative_images():
    """
    Images labeled blank with high model confidence
    """
    positive_image_subquery = positive_images().subquery()
    return (
        sa.select(m.Image)
        .join(m.ImageAnnotation)
        .join(m.Building, isouter=True)
        .where(m.Image.id.notin_(sa.select(positive_image_subquery.c.id)))
        .where(
            sa.and_(
                m.Image.bucket != "0",
                m.Image.bucket != "1",
                m.Image.bucket.is_not(None),
            )
        )
        .group_by(m.Image.id)
        .having(
            sa.func.sum(sa.cast(m.Building.id.is_(None), sa.Integer))
            == sa.func.count(m.ImageAnnotation.id)
        )
    )


def low_confidence_negative_images():
    """
    Images labeled blank with low model confidence
    """
    positive_image_subquery = positive_images().subquery()
    return (
        sa.select(m.Image)
        .join(m.ImageAnnotation)
        .join(m.Building, isouter=True)
        .where(m.Image.id.notin_(sa.select(positive_image_subquery.c.id)))
        .where(
            sa.or_(
                m.Image.bucket == "0",
                m.Image.bucket == "1",
            )
        )
        .group_by(m.Image.id)
        .having(
            sa.func.sum(sa.cast(m.Building.id.is_(None), sa.Integer))
            == sa.func.count(m.ImageAnnotation.id)
        )
    )


def ex_ante_labeled_images():
    return sa.select(m.Image).where(
        m.Image.bucket.is_not(None) & (m.Image.bucket != "0") & (m.Image.bucket != "1")
    )


def ex_ante_positive_images():
    return (
        sa.select(m.Image)
        .join(m.ImageAnnotation)
        .join(m.Building)
        .join(m.Facility)
        .join(m.CafoAnnotation, isouter=True)
        .group_by(m.Image.id)
        .where(m.Facility.archived_at.is_(None))
        .where(
            m.Image.bucket.is_not(None)
            & (m.Image.bucket != "0")
            & (m.Image.bucket != "1")
        )
        .having(
            (sa.func.count(m.CafoAnnotation.id) == 0)
            | (
                sa.func.sum(sa.cast(m.CafoAnnotation.is_cafo, sa.Integer))
                == sa.func.count(m.CafoAnnotation.id)
            )
        )
        .group_by(m.Facility.id)
    )


def adjacent_images(images_query):
    iq = images_query.subquery()
    image_alias = sa.orm.aliased(m.Image, name="image_alias")
    return (
        sa.select(m.Image)
        .join(
            image_alias,
            sa.and_(
                sa.func.ST_Intersects(
                    m.Image.geometry,
                    image_alias.geometry,
                ),
            ),
        )
        .where(
            sa.and_(
                m.Image.id != image_alias.id,
                image_alias.id.in_(sa.select(iq.c.id)),
                m.Image.bucket.is_not(None),
            )
        )
    )


def initially_labeled_images():
    ex_ante_labeled_images_query = ex_ante_labeled_images()
    adjacent_images_query = adjacent_images(ex_ante_positive_images())
    union_query = ex_ante_labeled_images_query.union(adjacent_images_query)
    return union_query


def initially_labeled_images_needing_labels():
    sq = initially_labeled_images().subquery()
    return (
        sa.select(m.Image)
        .join(m.ImageAnnotation, isouter=True)
        .where(m.ImageAnnotation.id.is_(None))
        .join(sq, sq.c.id == m.Image.id)
    )


def unlabeled_adjacent_images(images_query):
    ai = adjacent_images(images_query).subquery()
    return (
        sa.select(ai)
        .join(m.ImageAnnotation, isouter=True)
        .where(m.ImageAnnotation.id.is_(None))
    )


def unlabeled_images():
    return (
        sa.select(m.Image)
        .join(m.ImageAnnotation, isouter=True)
        .where(m.ImageAnnotation.id.is_(None) & m.Image.bucket.is_not(None))
    )


def labeled_images():
    return sa.select(m.Image).join(m.ImageAnnotation).where(m.Image.bucket.is_not(None))


def facility_expanded_permits(facility_ids=None):
    """
    Returns a query to get all expanded permit matches for facilities.
    If facility_ids is provided, only returns permits for those facilities.

    This efficiently matches permits that are within 1000 meters of facility geometries.
    """
    query = sa.select(m.Facility.id.label("facility_id"), m.Permit).join(
        m.Permit,
        sa.or_(
            sa.func.ST_DWithin(
                m.Permit.registered_location,
                m.Facility.geometry,
                1000,
            ),
            sa.func.ST_DWithin(
                m.Permit.geocoded_address_location,
                m.Facility.geometry,
                1000,
            ),
        ),
        isouter=True,
    )

    if facility_ids is not None:
        query = query.where(m.Facility.id.in_(facility_ids))

    return query
