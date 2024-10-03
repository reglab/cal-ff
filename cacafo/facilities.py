import datetime

import geoalchemy2 as ga
import rich.progress
import rich_click as click
import shapely as shp
import sqlalchemy as sa

import cacafo.db.sa_models as m
from cacafo.cluster.buildings import building_clusters
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


def join_facilities_to_permits(session=None):
    session = session or get_sqlalchemy_session()

    # Join facilities with permits based on parcels
    parcel_permit_joins = session.execute(
        sa.select(m.Facility, m.Permit)
        .join(m.Building, m.Facility.id == m.Building.facility_id)
        .join(
            m.Parcel,
            (m.Building.parcel_id == m.Parcel.id)
            & (
                (m.Permit.registered_location_parcel_id == m.Parcel.id)
                | (m.Permit.geocoded_address_location_parcel_id == m.Parcel.id)
            ),
        )
        .where(
            (m.Facility.archived_at.is_(None))
            & (m.Permit.facility_id.is_(None))
            & (m.Permit.registered_location_parcel_id.isnot(None))
            & (m.Permit.geocoded_address_location_parcel_id.isnot(None))
        )
        .group_by(m.Facility.id, m.Permit.id)
        .having(
            sa.func.count(
                sa.distinct(
                    sa.case(
                        (m.Permit.registered_location_parcel_id == m.Parcel.id, 1),
                        (
                            m.Permit.geocoded_address_location_parcel_id == m.Parcel.id,
                            2,
                        ),
                        else_=None,
                    )
                )
            )
            == 2
        )
    ).all()

    # Update facility_id for matched permits
    for facility, permit in parcel_permit_joins:
        permit.facility_id = facility.id
    session.add_all([permit for _, permit in parcel_permit_joins])
    session.flush()

    # Join remaining permits based on distance
    distance_permit_joins = session.execute(
        sa.select(m.Facility, m.Permit)
        .where(
            (m.Facility.archived_at.is_(None))
            & (m.Permit.facility_id.is_(None))
            & (m.Permit.registered_location.isnot(None))
            & (m.Permit.geocoded_address_location.isnot(None))
            & sa.func.ST_DWithin(m.Facility.geometry, m.Permit.registered_location, 200)
            & sa.func.ST_DWithin(
                m.Facility.geometry, m.Permit.geocoded_address_location, 200
            )
        )
        .group_by(m.Facility.id, m.Permit.id)
        .having(
            ~sa.exists().where(
                (m.Facility.id != m.Facility.id)
                & (
                    sa.func.ST_DWithin(
                        m.Facility.geometry, m.Permit.registered_location, 200
                    )
                    | sa.func.ST_DWithin(
                        m.Facility.geometry, m.Permit.geocoded_address_location, 200
                    )
                )
            )
        )
    ).all()

    # Update facility_id for distance-matched permits
    for facility, permit in distance_permit_joins:
        permit.facility_id = facility.id
    # Commit the changes
    session.flush()
    # Print summary
    click.secho(
        f"Joined {len(parcel_permit_joins)} permits to facilities based on parcels",
        fg="green",
    )
    click.secho(
        f"Joined {len(distance_permit_joins)} permits to facilities based on distance",
        fg="green",
    )


def join_facilities():
    session = get_sqlalchemy_session()
    # Join facilities with CafoAnnotations
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

    # Check for multiple facility mappings and update facility_id for CafoAnnotations
    cafo_annotation_facilities = {}
    for facility, cafo_annotation in cafo_joins:
        if cafo_annotation.id in cafo_annotation_facilities:
            raise ValueError(
                f"CafoAnnotation {cafo_annotation.id} maps to multiple facilities"
            )
        cafo_annotation_facilities[cafo_annotation.id] = facility.id
        cafo_annotation.facility_id = facility.id

    # Join facilities with AnimalTypeAnnotations
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

    # Check for multiple facility mappings and update facility_id for AnimalTypeAnnotations
    animal_type_annotation_facilities = {}
    for facility, animal_type_annotation in animal_type_joins:
        if animal_type_annotation.id in animal_type_annotation_facilities:
            raise ValueError(
                f"AnimalTypeAnnotation {animal_type_annotation.id} maps to multiple facilities"
            )
        animal_type_annotation_facilities[animal_type_annotation.id] = facility.id
        animal_type_annotation.facility_id = facility.id

    # Join facilities with ConstructionAnnotations
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

    # Check for multiple facility mappings and update facility_id for ConstructionAnnotations
    construction_annotation_facilities = {}
    for facility, construction_annotation in construction_joins:
        if construction_annotation.id in construction_annotation_facilities:
            raise ValueError(
                f"ConstructionAnnotation {construction_annotation.id} maps to multiple facilities"
            )
        construction_annotation_facilities[construction_annotation.id] = facility.id
        construction_annotation.facility_id = facility.id

    # Get the county for each facility
    facility_counties = session.execute(
        sa.select(m.Facility, m.County)
        .join(
            m.County,
            m.County.geometry.ST_Contains(
                sa.cast(m.Facility.geometry, ga.Geometry).ST_Centroid()
            ),
        )
        .where(m.Facility.archived_at.is_(None))
    ).all()

    # Update facility with county information
    for facility, county in facility_counties:
        facility.county_id = county.id

    # Commit the changes
    session.commit()

    # Print summary
    click.secho(f"Joined {len(cafo_joins)} CafoAnnotations to facilities", fg="green")
    click.secho(
        f"Joined {len(animal_type_joins)} AnimalTypeAnnotations to facilities",
        fg="green",
    )
    click.secho(
        f"Joined {len(construction_joins)} ConstructionAnnotations to facilities",
        fg="green",
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
