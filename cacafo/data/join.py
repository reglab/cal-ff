import typing as t
from dataclasses import dataclass

import rich_click as click
import sqlalchemy as sa
import sqlalchemy.orm as orm
from rich.traceback import install

import cacafo.db.sa_models as m
from cacafo.db.session import get_sqlalchemy_session

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
            session = get_sqlalchemy_session()
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


@click.command(
    "join", help="Fill in foreign key columns by joining data from other tables."
)
@click.option("--overwrite", is_flag=True)
@click.option("--add", is_flag=True)
@click.argument("tablename", type=click.Choice(Joiner.instances.keys()))
def _cli(tablename, overwrite, add):
    joiner = Joiner.instances[tablename]
    joiner.func(overwrite=overwrite, add=add)
