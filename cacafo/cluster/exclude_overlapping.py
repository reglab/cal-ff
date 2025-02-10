import datetime

import sqlalchemy as sa

from cacafo.db.models import Building
from cacafo.db.session import new_session
from cacafo.geom import mostly_overlapping_buildings


def exclude_overlapping_buildings():
    """
    Find all buildings that overlap >90% with another building and exclude all but the smallest.
    """
    session = new_session()

    # Get the building IDs to exclude from the geometry function
    building_ids_to_exclude = mostly_overlapping_buildings(session=session)

    # Update all the overlapping buildings to be excluded
    now = datetime.datetime.now()
    session.execute(
        sa.update(Building)
        .where(Building.id.in_(building_ids_to_exclude))
        .values(excluded_at=now, exclude_reason="overlaps >90% with another building")
    )

    # Commit the changes
    session.commit()

    return len(building_ids_to_exclude)


if __name__ == "__main__":
    num_excluded = exclude_overlapping_buildings()
    print(f"Excluded {num_excluded} overlapping buildings")
