import sqlalchemy as sa

import cacafo.db.sa_models as m


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
    return (
        sa.select(m.Image)
        .join(m.ImageAnnotation)
        .join(m.Building)
        .join(m.Facility)
        .join(m.CafoAnnotation, isouter=True)
        .group_by(m.Image.id)
        .where(m.Facility.archived_at.is_(None))
        .having(
            (sa.func.count(m.CafoAnnotation.id) == 0)
            | (
                sa.func.sum(sa.cast(m.CafoAnnotation.is_cafo, sa.Integer))
                == sa.func.count(m.CafoAnnotation.id)
            )
        )
        .group_by(m.Facility.id)
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
