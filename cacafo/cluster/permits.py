import sqlalchemy as sa
import geoalchemy2 as ga
import shapely as shp

from cacafo.db.session import new_session
from cacafo.transform import to_meters
import cacafo.db.models as m


def _conjunction_dict_of_sets(dos1, dos2):
    keys = set(dos1.keys()) | set(dos2.keys())
    return {k: dos1.get(k, set()) & dos2.get(k, set()) for k in keys}


def _disjunction_dict_of_sets(dos1, dos2):
    keys = set(dos1.keys()) | set(dos2.keys())
    return {k: dos1.get(k, set()) | dos2.get(k, set()) for k in keys}


def _invert_dict_of_sets(dos):
    sod = {}
    for k, v in dos.items():
        for i in v:
            sod[i] = sod.get(i, set()) | {k}
    return sod


def _remove_duplicate_entries(dos):
    """In a dict of sets, remove any set entries that are in another set entry"""
    sod = _invert_dict_of_sets(dos)
    dos_output = {}
    for s, d_set in sod.items():
        if len(d_set) == 1:
            d = d_set.pop()
            dos_output[d] = dos_output.get(d, set()) | {s}
    return dos_output


def _remove_empty_entries(dos):
    return {k: v for k, v in dos.items() if v}


def facility_permit_parcel_matches(session=None) -> dict[int, set[int]]:
    if session is None:
        session = new_session()
    GeocodedParcel = sa.orm.aliased(m.Parcel)
    RegisteredParcel = sa.orm.aliased(m.Parcel)
    GeocodedBuilding = sa.orm.aliased(m.Building)
    RegisteredBuilding = sa.orm.aliased(m.Building)
    GeocodedFacility = sa.orm.aliased(m.Facility)
    RegisteredFacility = sa.orm.aliased(m.Facility)

    parcel_permit_joins = (
        session.execute(
            sa.select(
                m.Permit.id.label("permit_id"),
                RegisteredFacility.id.label("facility_id"),
            )
            .join(
                RegisteredParcel,
                m.Permit.registered_location_parcel_id == RegisteredParcel.id,
            )
            .join(
                GeocodedParcel,
                m.Permit.geocoded_address_location_parcel_id == GeocodedParcel.id,
            )
            .join(
                RegisteredBuilding,
                RegisteredParcel.id == RegisteredBuilding.parcel_id,
            )
            .join(
                GeocodedBuilding,
                GeocodedParcel.id == GeocodedBuilding.parcel_id,
            )
            .join(
                RegisteredFacility,
                RegisteredBuilding.facility_id == RegisteredFacility.id,
            )
            .join(
                GeocodedFacility,
                GeocodedBuilding.facility_id == GeocodedFacility.id,
            )
            .where(
                (m.Permit.registered_location.isnot(None))
                & (m.Permit.geocoded_address_location.isnot(None))
                & (RegisteredFacility.id == GeocodedFacility.id)
                & (RegisteredFacility.archived_at.is_(None))
            )
            .distinct(m.Permit.id)
        )
        .mappings()
        .all()
    )
    facilities = {}
    for ppj in parcel_permit_joins:
        facilities[ppj["facility_id"]] = facilities.get(ppj["facility_id"], set()) | {
            ppj["permit_id"]
        }
    return facilities


def facility_permit_distance_matches(distance=200, excluded_facility_ids={}, session=None) -> dict[int, set[int]]:
    if session is None:
        session = new_session()
    facilities = list(
        session.execute(
            sa.select(m.Facility)
            .where(m.Facility.archived_at.is_(None))
            .where(m.Facility.id.notin_(excluded_facility_ids))
        )
        .scalars()
        .all()
    )
    permits = list(
        session.execute(sa.select(m.Permit).where(m.Permit.facility_id.is_(None)))
        .scalars()
        .all()
    )

    facility_geoms = [
        to_meters(ga.shape.to_shape(facility.geometry)) for facility in facilities
    ]
    permit_geocoded_address_geoms = [
        permit.geocoded_address_location
        and to_meters(ga.shape.to_shape(permit.geocoded_address_location))
        for permit in permits
    ]
    permit_registered_location_geoms = [
        permit.registered_location
        and to_meters(ga.shape.to_shape(permit.registered_location))
        for permit in permits
    ]

    facility_strtree = shp.STRtree(facility_geoms)
    # returns input indices, then tree indices
    geocoded_address_matches = facility_strtree.query(
        permit_geocoded_address_geoms,
        predicate="dwithin",
        distance=distance,
    )
    geocoded_matches = {}
    for permit_idx, facility_idx in zip(*geocoded_address_matches):
        geocoded_matches[permit_idx] = geocoded_matches.get(permit_idx, set()) | {
            facility_idx
        }
    registered_location_matches = facility_strtree.query(
        permit_registered_location_geoms,
        predicate="dwithin",
        distance=distance,
    )
    registered_matches = {}
    for permit_idx, facility_idx in zip(*registered_location_matches):
        registered_matches[permit_idx] = registered_matches.get(permit_idx, set()) | {
            facility_idx
        }

    both_matches = {
        k: v & geocoded_matches.get(k, set()) for k, v in registered_matches.items()
    }

    facility_to_permit_matches = {}
    for permit_idx, facilities in both_matches.items():
        if len(facilities) == 1:
            f = facilities.pop()
            facility_to_permit_matches[f] = facility_to_permit_matches.get(f, set()) | {
                permit_idx
            }
    return facility_to_permit_matches


def facility_parcel_then_distance_matches(distance=200, session=None) -> dict[int, set[int]]:
    """
    Returns a dictionary of facility ids to a set of permit ids, based on the
    criteria that both permit locations match to the same facility's parcels or within distance
    of the facility, and aren't within distance of any other facility.
    """
    if session is None:
        session = new_session()
    facility_parcel_matches = facility_permit_parcel_matches(session)
    facility_distance_matches = facility_permit_distance_matches(
        distance=distance, excluded_facility_ids=facility_parcel_matches.keys(), session=session
    )
    facility_matches = _disjunction_dict_of_sets(
        facility_parcel_matches, facility_distance_matches
    )
    # facility_matches = _remove_duplicate_entries(facility_matches)
    facility_matches = _remove_empty_entries(facility_matches)
    import ipdb; ipdb.set_trace()
    return facility_matches
