import datetime

import geoalchemy2 as ga
import pyproj
import rich.progress
import rich_click as click
import shapely as shp
import shapely.ops
import sqlalchemy as sa

import cacafo.db.sa_models as m
from cacafo.cluster.buildings import building_clusters
from cacafo.constants import CA_SRID, DEFAULT_SRID
from cacafo.db.session import get_sqlalchemy_session


def create_facilities():
    session = get_sqlalchemy_session()
    session.execute(sa.update(m.Facility).values(archived_at=datetime.datetime.now()))
    session.flush()
    bc = building_clusters()
    to_create = []
    buildings_to_update = []
    building_ids_map = {
        building.id: {
            "geometry": ga.shape.to_shape(building.geometry),
            "building": building,
        }
        for building in session.execute(sa.select(m.Building)).scalars().all()
    }
    for cluster in rich.progress.track(list(bc), description="Creating facilities"):
        geom = shp.geometry.MultiPolygon(
            [
                building_ids_map[building_id]["geometry"]
                for building_id in cluster
                if building_id in building_ids_map
            ]
        )
        facility = m.Facility(
            archived_at=None,
            geometry=geom.wkt,
        )
        for building_id in cluster:
            if building_id in building_ids_map:
                building = building_ids_map[building_id]["building"]
                building.facility = facility
                buildings_to_update.append(building)
        to_create.append(facility)
    session.add_all(buildings_to_update)
    session.add_all(to_create)
    session.commit()


def join_facilities():
    session = get_sqlalchemy_session()
    join_cafo_annotations(session)
    join_animal_type_annotations(session)
    join_construction_annotations(session)
    join_facility_counties(session)
    join_permits(session)
    session.commit()


def join_permits(session=None):
    session = session or get_sqlalchemy_session()

    # Join facilities with permits based on parcels
    GeocodedParcel = sa.orm.aliased(m.Parcel)
    RegisteredParcel = sa.orm.aliased(m.Parcel)
    GeocodedBuilding = sa.orm.aliased(m.Building)
    RegisteredBuilding = sa.orm.aliased(m.Building)
    GeocodedFacility = sa.orm.aliased(m.Facility)
    RegisteredFacility = sa.orm.aliased(m.Facility)
    parcel_permit_joins = (
        session.execute(
            sa.select(m.Permit)
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
            )
            .distinct(m.Permit.id)
        )
        .scalars()
        .all()
    )

    # Update facility_id for matched permits
    for permit in parcel_permit_joins:
        permit.facility_id = permit.registered_location_parcel.buildings[0].facility_id
    session.add_all(parcel_permit_joins)
    session.flush()

    # make a strtree of facilities
    facilities = list(
        session.execute(sa.select(m.Facility).where(m.Facility.archived_at.is_(None)))
        .scalars()
        .all()
    )
    permits = list(
        session.execute(sa.select(m.Permit).where(m.Permit.facility_id.is_(None)))
        .scalars()
        .all()
    )

    to_meters = pyproj.Transformer.from_crs(DEFAULT_SRID, CA_SRID, always_xy=True)

    def g2m(geom):
        return shapely.ops.transform(to_meters.transform, geom)

    facility_geoms = [
        g2m(ga.shape.to_shape(facility.geometry)) for facility in facilities
    ]
    permit_geocoded_address_geoms = [
        permit.geocoded_address_location
        and g2m(ga.shape.to_shape(permit.geocoded_address_location))
        for permit in permits
    ]
    permit_registered_location_geoms = [
        permit.registered_location
        and g2m(ga.shape.to_shape(permit.registered_location))
        for permit in permits
    ]

    facility_strtree = shp.STRtree(facility_geoms)
    # returns input indices, then tree indices
    geocoded_address_matches = facility_strtree.query(
        permit_geocoded_address_geoms,
        predicate="dwithin",
        distance=200,
    )
    geocoded_matches = {}
    for permit_idx, facility_idx in zip(*geocoded_address_matches):
        geocoded_matches[permit_idx] = geocoded_matches.get(permit_idx, set()) | {
            facility_idx
        }
    registered_location_matches = facility_strtree.query(
        permit_registered_location_geoms,
        predicate="dwithin",
        distance=200,
    )
    registered_matches = {}
    for permit_idx, facility_idx in zip(*registered_location_matches):
        registered_matches[permit_idx] = registered_matches.get(permit_idx, set()) | {
            facility_idx
        }

    both_matches = {
        k: v & geocoded_matches.get(k, set()) for k, v in registered_matches.items()
    }
    permits_to_update = []
    for permit_id, matched_facilities in both_matches.items():
        if len(matched_facilities) == 1:
            permit = permits[permit_id]
            facility = facilities[matched_facilities.pop()]
            permit.facility_id = facility.id
            permits_to_update.append(permit)
    session.add_all(permits_to_update)
    session.flush()
    session.commit()
    # Print summary
    click.secho(
        f"Joined {len(parcel_permit_joins)} permits to facilities based on parcels",
        fg="green",
    )
    click.secho(
        f"Joined {len(permits_to_update)} permits to facilities based on distance",
        fg="green",
    )


def join_cafo_annotations(session):
    session.execute(sa.update(m.CafoAnnotation).values(facility_id=None))
    cafo_joins = session.execute(
        sa.select(m.Facility, m.CafoAnnotation)
        .join(
            m.CafoAnnotation,
            sa.cast(m.Facility.geometry, ga.Geometry)
            .ST_Envelope()
            .ST_Contains(sa.cast(m.CafoAnnotation.location, ga.Geometry)),
        )
        .where(m.Facility.archived_at.is_(None))
    ).all()

    cafo_annotation_facilities = {}
    for facility, cafo_annotation in rich.progress.track(
        cafo_joins, description="Processing CafoAnnotations"
    ):
        if (
            cafo_annotation.id in cafo_annotation_facilities
            and ga.shape.to_shape(facility.geometry).area
            > ga.shape.to_shape(
                cafo_annotation_facilities[cafo_annotation.id].geometry
            ).area
        ):
            continue
        cafo_annotation_facilities[cafo_annotation.id] = facility
        cafo_annotation.facility_id = facility.id

    session.flush()
    session.commit()
    click.secho(f"Joined {len(cafo_joins)} CafoAnnotations to facilities", fg="green")


def join_animal_type_annotations(session):
    session.execute(sa.update(m.AnimalTypeAnnotation).values(facility_id=None))
    animal_type_joins = session.execute(
        sa.select(m.Facility, m.AnimalTypeAnnotation)
        .join(
            m.AnimalTypeAnnotation,
            sa.cast(m.Facility.geometry, ga.Geometry)
            .ST_Envelope()
            .ST_Contains(sa.cast(m.AnimalTypeAnnotation.location, ga.Geometry)),
        )
        .where(m.Facility.archived_at.is_(None))
    ).all()

    animal_type_annotation_facilities = {}
    for facility, animal_type_annotation in rich.progress.track(
        animal_type_joins, description="Processing AnimalTypeAnnotations"
    ):
        if (
            animal_type_annotation.id in animal_type_annotation_facilities
            and ga.shape.to_shape(facility.geometry).area
            > ga.shape.to_shape(
                animal_type_annotation_facilities[animal_type_annotation.id].geometry
            ).area
        ):
            continue
        animal_type_annotation_facilities[animal_type_annotation.id] = facility
        animal_type_annotation.facility_id = facility.id

    session.flush()
    session.commit()
    click.secho(
        f"Joined {len(animal_type_joins)} AnimalTypeAnnotations to facilities",
        fg="green",
    )


def join_construction_annotations(session):
    session.execute(sa.update(m.ConstructionAnnotation).values(facility_id=None))
    construction_joins = session.execute(
        sa.select(m.Facility, m.ConstructionAnnotation)
        .join(
            m.ConstructionAnnotation,
            sa.cast(m.Facility.geometry, ga.Geometry)
            .ST_Envelope()
            .ST_Contains(sa.cast(m.ConstructionAnnotation.location, ga.Geometry)),
        )
        .where(m.Facility.archived_at.is_(None))
    ).all()

    construction_annotation_facilities = {}
    for facility, construction_annotation in rich.progress.track(
        construction_joins, description="Processing ConstructionAnnotations"
    ):
        if (
            construction_annotation.id in construction_annotation_facilities
            and ga.shape.to_shape(facility.geometry).area
            > ga.shape.to_shape(
                construction_annotation_facilities[construction_annotation.id].geometry
            ).area
        ):
            continue
        construction_annotation_facilities[construction_annotation.id] = facility
        construction_annotation.facility_id = facility.id

    click.secho(
        f"Joined {len(construction_joins)} ConstructionAnnotations to facilities",
        fg="green",
    )


def join_facility_counties(session):
    facility_counties = session.execute(
        sa.select(m.Facility, m.Parcel.county_id)
        .join(m.Building, m.Facility.id == m.Building.facility_id)
        .join(m.Parcel, m.Building.parcel_id == m.Parcel.id)
        .where(m.Facility.archived_at.is_(None))
    ).all()

    county_id_map = {
        county.id: county
        for county in session.execute(sa.select(m.County)).scalars().all()
    }

    facility_counties_dict = {}
    for facility, county_id in rich.progress.track(
        facility_counties, description="Processing facility counties"
    ):
        if facility.id in facility_counties_dict:
            facility_counties_dict[facility.id][county_id] = (
                facility_counties_dict[facility.id].get(county_id, 0) + 1
            )
        else:
            facility_counties_dict[facility.id] = {county_id: 1}

    for facility_id, counties in facility_counties_dict.items():
        county_id = max(counties, key=counties.get)
        session.execute(
            sa.update(m.Facility)
            .where(m.Facility.id == facility_id)
            .values(county_id=county_id_map[county_id].id)
        )


@click.group("facilities")
def _cli():
    """Commands for managing facilities"""
    pass


@_cli.command("create", help="Create facilities from building clusters")
def create_facilities_cli():
    create_facilities()
    # count number of facilities
    session = get_sqlalchemy_session()
    count = session.execute(
        sa.select(sa.func.count(m.Facility.id)).where(m.Facility.archived_at.is_(None))
    ).scalar()
    click.secho(f"Created {count} facilities", fg="green")


@_cli.command("join", help="Join annotations to facilities")
def join_facilities_cli():
    join_facilities()
    click.echo("Joined annotations to facilities.")


@_cli.command("archive", help="Archive facilities")
def archive_facilities_cli():
    click.confirm("Are you sure you want to archive all facilities?", abort=True)
    session = get_sqlalchemy_session()
    archived_at = datetime.datetime.now()
    session.execute(
        sa.update(m.Facility)
        .values(archived_at=archived_at)
        .where(m.Facility.archived_at.is_(None))
    )
    session.flush()
    count = session.execute(
        sa.select(sa.func.count(m.Facility.id)).where(
            m.Facility.archived_at == archived_at
        )
    ).scalar()
    click.secho(f"Archived {count} facilities", fg="green")
