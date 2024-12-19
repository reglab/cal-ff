import sqlalchemy as sa

import cacafo.db.models as m


def cafos():
    return (
        sa.select(m.Facility)
        .join(m.CafoAnnotation, isouter=True)
        .group_by(m.Facility.id)
        .having(
            (
                sa.func.sum(sa.cast(m.CafoAnnotation.is_cafo, sa.Integer))
                == sa.func.count(m.CafoAnnotation.id)
            )
            | (sa.func.count(m.CafoAnnotation.id) == 0)
        )
        .where(m.Facility.archived_at.is_(None))
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
