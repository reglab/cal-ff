import geoalchemy2 as ga
import numpy as np
import rich.progress
import rich_click as click
import shapely as shp
import sqlalchemy as sa

import cacafo.data.source
import cacafo.db.models as m
import cacafo.owner_name_matching
from cacafo.db.session import new_session
from cacafo.transform import CA_SRID

BUILDING_RELATIONSHIP_TYPES = {}


def relationship_type(name: str):
    def decorator(cls):
        BUILDING_RELATIONSHIP_TYPES[name] = cls
        return cls

    return decorator


@relationship_type("distance")
def add_distance_relationships(session):
    buildings = session.execute(
        sa.select(
            m.Building.id.label("building_id"),
            sa.func.ST_Transform(
                sa.cast(m.Building.geometry, ga.Geometry),
                CA_SRID,
            ).label("geometry"),
        ).order_by(m.Building.id)
    ).all()
    building_ids = [b[0] for b in buildings]
    geometries = [ga.shape.to_shape(b[1]) for b in buildings]
    tree = shp.STRtree(geometries)
    input_idxs, tree_idxs = tree.query(geometries, predicate="dwithin", distance=1000)
    distances = np.array(
        [geometries[i].distance(geometries[j]) for i, j in zip(input_idxs, tree_idxs)]
    )
    to_create = []
    for input_idx, tree_idx, distance in rich.progress.track(
        zip(input_idxs, tree_idxs, distances),
        description="Building distance relationships",
        total=len(input_idxs),
    ):
        if input_idx == tree_idx:
            continue
        to_create.append(
            m.BuildingRelationship(
                building_id=building_ids[input_idx],
                related_building_id=building_ids[tree_idx],
                reason="distance",
                weight=1000 - int(distance),
            )
        )
        if len(to_create) > 1000:
            session.add_all(to_create)
            session.flush()
            to_create = []
    session.add_all(to_create)
    session.flush()


@relationship_type("all")
def add_all_relationships(session):
    add_distance_relationships(session)
    add_matching_parcel_relationships(session)
    add_tfidf_relationships(session)
    add_fuzzy_relationships(session)
    add_parcel_owner_annotation_relationships(session)


@relationship_type("tfidf")
def add_tfidf_relationships(session):
    building_relationships = (
        session.execute(
            sa.select(m.BuildingRelationship).where(
                m.BuildingRelationship.reason == "distance"
            )
        )
        .scalars()
        .all()
    )

    building_id_to_parcel_owner_name = dict(
        session.execute(
            sa.select(m.Building.id, m.Parcel.owner)
            .join(m.Parcel)
            .where(m.Building.parcel_id.is_not(None))
        ).all()
    )

    all_owner_names = set(building_id_to_parcel_owner_name.values())

    to_create = []
    for building_relationship in rich.progress.track(
        building_relationships, description="Building tfidf relationships"
    ):
        if (
            building_relationship.building_id not in building_id_to_parcel_owner_name
            or building_relationship.related_building_id
            not in building_id_to_parcel_owner_name
        ):
            continue

        weight = cacafo.owner_name_matching.tf_idf(
            all_owner_names,
            building_id_to_parcel_owner_name[building_relationship.building_id],
            building_id_to_parcel_owner_name[building_relationship.related_building_id],
        )
        to_create.append(
            m.BuildingRelationship(
                building_id=building_relationship.building_id,
                related_building_id=building_relationship.related_building_id,
                reason="tfidf",
                weight=weight,
            )
        )
        if len(to_create) > 1000:
            session.add_all(to_create)
            session.flush()
            to_create = []

    session.add_all(to_create)
    session.commit()


@relationship_type("fuzzy")
def add_fuzzy_relationships(session):
    building_relationships = (
        session.execute(
            sa.select(m.BuildingRelationship).where(
                m.BuildingRelationship.reason == "distance"
            )
        )
        .scalars()
        .all()
    )

    building_id_to_parcel_owner_name = dict(
        session.execute(
            sa.select(m.Building.id, m.Parcel.owner)
            .join(m.Parcel)
            .where(m.Building.parcel_id.is_not(None))
        ).all()
    )

    to_create = []
    for building_relationship in rich.progress.track(
        building_relationships, description="Building fuzzy relationships"
    ):
        if (
            building_relationship.building_id not in building_id_to_parcel_owner_name
            or building_relationship.related_building_id
            not in building_id_to_parcel_owner_name
        ):
            continue

        to_create.append(
            m.BuildingRelationship(
                building_id=building_relationship.building_id,
                related_building_id=building_relationship.related_building_id,
                reason="fuzzy",
                weight=cacafo.owner_name_matching.fuzzy(
                    building_id_to_parcel_owner_name[building_relationship.building_id],
                    building_id_to_parcel_owner_name[
                        building_relationship.related_building_id
                    ],
                ),
            )
        )
        if len(to_create) > 1000:
            session.add_all(to_create)
            session.flush()
            to_create = []

    session.add_all(to_create)
    session.commit()


@relationship_type("parcel_owner_annotation")
def add_parcel_owner_annotation_relationships(session):
    building_relationships = (
        session.execute(
            sa.select(m.BuildingRelationship).where(
                m.BuildingRelationship.reason == "distance"
            )
        )
        .scalars()
        .all()
    )

    building_id_to_parcel_owner_name = dict(
        session.execute(
            sa.select(m.Building.id, m.Parcel.owner)
            .join(m.Parcel)
            .where(m.Building.parcel_id.is_not(None))
        ).all()
    )

    to_create = []
    for building_relationship in rich.progress.track(
        building_relationships,
        description="Building parcel owner annotation relationships",
    ):
        if (
            building_relationship.building_id not in building_id_to_parcel_owner_name
            or building_relationship.related_building_id
            not in building_id_to_parcel_owner_name
        ):
            continue

        to_create.append(
            m.BuildingRelationship(
                building_id=building_relationship.building_id,
                related_building_id=building_relationship.related_building_id,
                reason="parcel owner annotation",
                weight=cacafo.owner_name_matching.annotation(
                    building_id_to_parcel_owner_name[building_relationship.building_id],
                    building_id_to_parcel_owner_name[
                        building_relationship.related_building_id
                    ],
                ),
            )
        )
        if len(to_create) > 1000:
            session.add_all(to_create)
            session.flush()
            to_create = []

    session.add_all(to_create)
    session.commit()


@relationship_type("matching_parcel")
def add_matching_parcel_relationships(session):
    # Get all buildings with their parcel_ids
    buildings = session.execute(
        sa.select(m.Building.id, m.Building.parcel_id)
        .where(m.Building.parcel_id.is_not(None))
        .order_by(m.Building.parcel_id)
    ).all()

    # Group buildings by parcel_id
    parcel_groups = {}
    for building_id, parcel_id in buildings:
        parcel_groups.setdefault(parcel_id, []).append(building_id)

    to_create = []
    for parcel_id, building_ids in rich.progress.track(
        parcel_groups.items(),
        description="Building matching parcel relationships",
        total=len(parcel_groups),
    ):
        # Create relationships between all buildings in the same parcel
        for i, building_id in enumerate(building_ids):
            for related_building_id in building_ids[i + 1 :]:
                # Create relationship from building_id to related_building_id
                to_create.append(
                    m.BuildingRelationship(
                        building_id=building_id,
                        related_building_id=related_building_id,
                        reason="matching parcel",
                        weight=1000,  # You can adjust this weight
                    )
                )
                # Create relationship from related_building_id to building_id
                to_create.append(
                    m.BuildingRelationship(
                        building_id=related_building_id,
                        related_building_id=building_id,
                        reason="matching parcel",
                        weight=1000,  # You can adjust this weight as needed
                    )
                )

        if len(to_create) > 1000:
            session.add_all(to_create)
            session.flush()
            to_create = []

    session.add_all(to_create)
    session.commit()


@click.group("buildrel")
def _cli():
    """Commands for managing facilities"""
    pass


@_cli.command("create", help="Create building relationships")
@click.option(
    "--type",
    type=click.Choice(BUILDING_RELATIONSHIP_TYPES.keys()),
    help="Type of building relationships to create",
)
def create_building_relationships_cli(type: str):
    session = new_session()
    if type is None:
        add_all_relationships(session)
    else:
        BUILDING_RELATIONSHIP_TYPES[type](session)
    session.commit()
    num_created = session.execute(
        sa.select(sa.func.count(m.BuildingRelationship.id))
    ).scalar()
    click.secho(f"Created {num_created} building relationships", fg="green")


@_cli.command("delete", help="Delete building relationships")
@click.option(
    "--type",
    type=click.Choice(BUILDING_RELATIONSHIP_TYPES.keys()),
    help="Type of building relationships to delete",
    default="all",
)
def delete_building_relationships_cli(type: str):
    session = new_session()
    condition = True
    if type != "all":
        condition = m.BuildingRelationship.reason == type.replace("_", " ")
    num_deleted = session.execute(
        sa.delete(m.BuildingRelationship).where(condition)
    ).rowcount
    session.commit()
    click.secho(f"Deleted {num_deleted} building relationships", fg="green")
