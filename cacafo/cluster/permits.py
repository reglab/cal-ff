from cacafo.db.models import (
    Building,
    FacilityPermittedLocation,
    Permit,
    PermitPermittedLocation,
    PermittedLocation,
)


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


def facility_permit_distance_matches(facility_permitted_location_conditions):
    fpls = (
        FacilityPermittedLocation.select(
            FacilityPermittedLocation.facility_id.alias("facility_id"),
            PermitPermittedLocation.permit_id.alias("permit_id"),
        )
        .join(PermittedLocation)
        .join(PermitPermittedLocation)
        .where(facility_permitted_location_conditions)
        .dicts()
    )
    facilities = {}
    for fpl in fpls:
        facilities[fpl["facility_id"]] = facilities.get(fpl["facility_id"], set()) | {
            fpl["permit_id"]
        }
    return facilities


def facility_permit_parcel_matches():
    facility_permits = (
        Building.select(
            Building.facility_id.alias("facility_id"),
            Permit.id.alias("permit_id"),
        )
        .join(Permit, on=(Building.parcel_id == Permit.parcel_id))
        .dicts()
    )
    facilities = {}
    for fp in facility_permits:
        facilities[fp["facility_id"]] = facilities.get(fp["facility_id"], set()) | {
            fp["permit_id"]
        }
    return facilities


def parcel_then_distance_matches(distance=200, cow_only=False) -> dict[int, set[int]]:
    """
    Returns a dictionary of facility ids to a set of permit ids, based on the
    criteria that both permit locations match to the same facility's parcels or within distance
    of the facility, and aren't within distance of any other facility.
    """
    permit_data_filter = PermitPermittedLocation.source == "permit data"
    geocoding_filter = PermitPermittedLocation.source == "address geocoding"
    distance_filter = lambda d: FacilityPermittedLocation.distance < d

    parcel_matches = facility_permit_parcel_matches()
    permit_matches = facility_permit_distance_matches(
        permit_data_filter & distance_filter(200)
    )
    geocoded_matches = facility_permit_distance_matches(
        geocoding_filter & distance_filter(200)
    )

    distance_matches = _conjunction_dict_of_sets(permit_matches, geocoded_matches)

    matches = _disjunction_dict_of_sets(
        parcel_matches,
        _remove_empty_entries(_remove_duplicate_entries(distance_matches)),
    )
    cow_permit_ids = {
        p.id
        for p in Permit.select(Permit.id).where(Permit.data["Program"] == "ANIWSTCOWS")
    }
    cow_permit_matches = lambda matches: _remove_empty_entries(
        {
            f: {p for p in permits if p in cow_permit_ids}
            for f, permits in matches.items()
        }
    )
    return matches if not cow_only else cow_permit_matches(matches)
