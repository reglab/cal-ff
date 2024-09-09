import csv
import typing as t
from dataclasses import dataclass

import rich.progress
import rich_click as click
import sqlalchemy as sa
from rich.traceback import install

import cacafo.data.source
import cacafo.db.sa_models as m
from cacafo.db.session import get_sqlalchemy_session

install(show_locals=True)


def _is_populated(session, model):
    return session.execute(sa.select(sa.func.count()).select_from(model)).scalar() > 0


@dataclass
class Ingestor:
    model: m.Base
    func: t.Callable
    depends_on: t.List[m.Base]

    instances: t.ClassVar[dict[str, "Ingestor"]] = {}


def _get_dependencies(model: m.Base) -> t.List[m.Base]:
    immediate_dependences = Ingestor.instances[model.__tablename__].depends_on
    return immediate_dependences + sum(
        [_get_dependencies(dep) for dep in immediate_dependences], []
    )


def _preflight(session, model, overwrite=False, add=False):
    dependencies = _get_dependencies(model)
    for dep in dependencies:
        if not _is_populated(session, dep):
            raise ValueError(
                f"Table {model.__name__} depends on {dep.__name__}, which is not populated."
            )
    if add:
        return
    if overwrite:
        session.execute(sa.delete(model))
        session.commit()
        return
    if _is_populated(session, model):
        raise ValueError(
            f"Table {model.__name__} is already populated;"
            "pass `overwrite=True` to wipe and repopulate, and `add=True`"
            "to add to existing data."
        )


def ingestor(model, depends_on=[]):
    def decorator(func):
        def wrapper(overwrite=False, add=False):
            session = get_sqlalchemy_session()
            _preflight(session, model, overwrite, add)
            previous_num = session.execute(
                sa.select(sa.func.count()).select_from(model)
            ).scalar()
            func(session)
            session.commit()
            post_num = session.execute(
                sa.select(sa.func.count()).select_from(model)
            ).scalar()
            click.secho(
                f"Ingested {post_num - previous_num} rows into {model.__tablename__}",
                fg="green",
            )

        Ingestor.instances[model.__tablename__] = Ingestor(model, wrapper, depends_on)
        return wrapper

    return decorator


@ingestor(m.CountyGroup)
def county_group(session):
    with open(cacafo.data.source.get("county_groups.csv")) as f:
        reader = csv.DictReader(f)
        county_groups = []
        for line in rich.progress.track(reader, description="Ingesting county groups"):
            county_groups.append(m.CountyGroup(name=line["Group Name"]))
        session.add_all(county_groups)


@ingestor(m.County, depends_on=[m.CountyGroup])
def county(session):
    county_group_name_to_id = {
        name: id
        for name, id in session.execute(
            sa.select(m.CountyGroup.name, m.CountyGroup.id)
        ).all()
    }

    with open(cacafo.data.source.get("county_groups.csv")) as f:
        reader = csv.DictReader(f)
        county_name_to_group_id = {
            line["County"]: county_group_name_to_id[line["Group Name"]]
            for line in reader
        }

    with open(cacafo.data.source.get("counties.csv")) as f:
        csv.field_size_limit(1000000)
        reader = csv.DictReader(f)
        counties = []
        for line in rich.progress.track(reader, description="Ingesting counties"):
            counties.append(
                m.County(
                    name=line["Name"],
                    geometry=line["the_geom"],
                    county_group_id=county_name_to_group_id[line["Name"]],
                )
            )
        session.add_all(counties)


@click.command("ingest", help="Ingest data into the database")
@click.option("--overwrite", is_flag=True)
@click.option("--add", is_flag=True)
@click.argument("tablename", type=click.Choice(Ingestor.instances.keys()))
def _cli(tablename, overwrite, add):
    ingestor = Ingestor.instances[tablename]
    ingestor.func(overwrite=overwrite, add=add)
