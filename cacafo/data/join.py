import typing as t
from dataclasses import dataclass

import rich
import rich.prompt
import rich_click as click
import sqlalchemy as sa
import sqlalchemy.orm as orm
from rich.traceback import install

import cacafo.db.models as m
from cacafo.db.session import new_session

install(show_locals=True)

DEFAULT_EPSG = 4326
CA_EPSG = 3311


def _is_populated(session, model):
    return session.execute(sa.select(sa.func.count()).select_from(model)).scalar() > 0


def _column_is_populated(session, column):
    model = column.table
    return (
        session.execute(
            sa.select(sa.func.count()).select_from(model).where(column.isnot(None))
        ).scalar()
        > 0
    )


@dataclass
class Joiner:
    column: orm.attributes.InstrumentedAttribute
    func: t.Callable
    depends_on: t.List[m.Base]

    instances: t.ClassVar[dict[str, "Joiner"]] = {}


def _preflight(session, column, extra_dependencies=[], overwrite=False, add=False):
    for dep in (
        extra_dependencies
        + [column.table]
        + [fk.column.table for fk in column.foreign_keys]
    ):
        if not _is_populated(session, dep):
            raise ValueError(
                f"Foreign key {column} depends on table {dep.__name__}, which is not populated."
            )
    if overwrite:
        session.execute(column.table.update().values({column: None}))
        return
    if add:
        return
    if _column_is_populated(session, column):
        raise ValueError(
            f"Column {column} is already populated. Pass --overwrite to overwrite, or --add to fill in values."
        )


def joiner(column, extra_dependencies=[]):
    def decorator(func):
        def wrapper(overwrite=False, add=False):
            session = new_session()
            _preflight(session, column, extra_dependencies, overwrite, add)
            previous_num = session.execute(
                sa.select(sa.func.count())
                .select_from(column.table)
                .where(column.isnot(None))
            ).scalar()
            func(session)
            session.commit()
            post_num = session.execute(
                sa.select(sa.func.count())
                .select_from(column.table)
                .where(column.isnot(None))
            ).scalar()
            remaining_num = session.execute(
                sa.select(sa.func.count())
                .select_from(column.table)
                .where(column.is_(None))
            ).scalar()
            click.secho(
                f"Filled in {post_num - previous_num} values for column {column}.",
                fg="green",
            )
            if remaining_num > 0:
                click.secho(
                    f"{remaining_num} rows still have NULL values in column {column}.",
                    fg="yellow",
                )

        name = f"{column.table.name}-{column.name}"
        Joiner.instances[name] = Joiner(column, wrapper, extra_dependencies)
        return wrapper

    return decorator


@joiner(m.Permit.geocoded_address_location_parcel_id)
def permit_geocoded_address_location_parcel_id(session):
    click.secho("Joining permits to parcels by geocoded address location...", fg="blue")
    query = (
        sa.select(m.Permit.id, m.Parcel.id)
        .select_from(m.Permit)
        .join(
            m.Parcel,
            m.Permit.geocoded_address_location.ST_Intersects(
                m.Parcel.inferred_geometry
            ),
        )
        .where(
            m.Permit.geocoded_address_location.isnot(None)
            & m.Permit.geocoded_address_location_parcel_id.is_(None)
        )
    )
    values = [
        {
            "id": permit_id,
            "geocoded_address_location_parcel_id": parcel_id,
        }
        for permit_id, parcel_id in session.execute(query)
    ]
    click.secho(f"Updating {len(values)} rows...", fg="blue")
    session.execute(
        sa.update(m.Permit),
        values,
    )


@joiner(m.Permit.registered_location_parcel_id)
def permit_registered_location_parcel_id(session):
    click.secho("Joining permits to parcels by registered location...", fg="blue")
    query = (
        sa.select(m.Permit.id, m.Parcel.id)
        .select_from(m.Permit)
        .join(
            m.Parcel,
            m.Permit.registered_location.ST_Intersects(m.Parcel.inferred_geometry),
        )
        .where(
            m.Permit.registered_location.isnot(None)
            & m.Permit.registered_location_parcel_id.is_(None)
        )
    )
    values = [
        {
            "id": permit_id,
            "registered_location_parcel_id": parcel_id,
        }
        for permit_id, parcel_id in session.execute(query)
    ]
    click.secho(f"Updating {len(values)} rows...", fg="blue")
    session.execute(
        sa.update(m.Permit),
        values,
    )


@joiner(m.ImageAnnotation.image_id)
def image_annotation_image_id(session):
    click.secho("Joining image annotations to images...", fg="blue")
    query = (
        sa.select(m.ImageAnnotation.id, m.ImageAnnotation.data["filename"])
        .select_from(m.ImageAnnotation)
        .where(m.ImageAnnotation.image_id.is_(None))
    )
    image_name_to_id = {
        name: id
        for id, name in session.execute(
            sa.select(m.Image.id, m.Image.name).select_from(m.Image)
        ).fetchall()
    }
    values = [
        {
            "id": annotation_id,
            "image_id": image_name_to_id[filename.split("/")[-1].replace(".jpeg", "")],
        }
        for annotation_id, filename in session.execute(query)
    ]
    click.secho(f"Updating {len(values)} rows...", fg="blue")
    session.execute(
        sa.update(m.ImageAnnotation),
        values,
    )


@joiner(m.Building.parcel_id)
def building_parcel_id(session):
    click.secho("Joining buildings to parcels...", fg="blue")
    query = (
        sa.select(m.Building.id, m.Parcel.id)
        .select_from(m.Building)
        .join(
            m.Parcel,
            m.Building.geometry.ST_Intersects(m.Parcel.inferred_geometry),
        )
        .where(m.Building.parcel_id.is_(None))
    )
    values = [
        {
            "id": building_id,
            "parcel_id": parcel_id,
        }
        for building_id, parcel_id in session.execute(query)
    ]
    click.secho(f"Updating {len(values)} rows...", fg="blue")
    session.execute(
        sa.update(m.Building),
        values,
    )
    unjoined_count = session.execute(
        sa.select(sa.func.count())
        .select_from(m.Building)
        .where(m.Building.parcel_id.is_(None))
    ).scalar()
    if unjoined_count > 0:
        click.secho(
            f"{unjoined_count} buildings could not be joined to parcels.", fg="yellow"
        )


@joiner(m.Building.census_block_id)
def building_census_block_id(session):
    click.secho("Joining buildings to parcels...", fg="blue")
    query = (
        sa.select(m.Building.id, m.CensusBlock.id)
        .select_from(m.Building)
        .join(
            m.CensusBlock,
            m.Building.geometry.ST_Intersects(m.CensusBlock.geometry),
        )
        .group_by(m.Building.id)
        .where(m.Building.census_block_id.is_(None))
        .having(sa.func.count(m.CensusBlock.id) == 1)
    )
    values = [
        {
            "id": building_id,
            "census_block_id": census_block_id,
        }
        for building_id, census_block_id in session.execute(query)
    ]
    click.secho(f"Updating {len(values)} rows...", fg="blue")
    session.execute(
        sa.update(m.Building),
        values,
    )
    unjoined_count = session.execute(
        sa.select(sa.func.count())
        .select_from(m.Building)
        .where(m.Building.census_block_id.is_(None))
    ).scalar()
    if unjoined_count > 0:
        click.secho(
            f"{unjoined_count} buildings could not be joined to parcels.", fg="yellow"
        )


@click.command(
    "join", help="Fill in foreign key columns by joining data from other tables."
)
@click.option("--overwrite", is_flag=True)
@click.option("--add", is_flag=True)
@click.option("--delete", is_flag=True)
@click.argument("tablename", type=click.Choice(Joiner.instances.keys()))
def _cli(tablename, overwrite, add, delete):
    joiner = Joiner.instances[tablename]
    if delete:
        # confirm
        conf = rich.prompt.Confirm.ask(
            f"Are you sure you want to delete all values in column {joiner.column}?"
        )
        if not conf:
            rich.print("Aborting.")
            return
        session = new_session()
        session.execute(joiner.column.table.update().values({joiner.column: None}))
        session.commit()
        return
    joiner.func(overwrite=overwrite, add=add)
