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


def initially_labeled_images():
    return sa.select(m.Image).where(
        m.Image.bucket.is_not(None) & (m.Image.bucket != "0") & (m.Image.bucket != "1")
    )


def initial_positives():
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
