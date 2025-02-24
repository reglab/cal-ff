import logging
import subprocess
import sys
import traceback
from pathlib import Path

import ipdb
import rich_click as click
from IPython import start_ipython
from rich.console import Console
from rich.traceback import Traceback
from rl.utils.logger import LOGGER


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    help="Set the log level",
    type=click.Choice(list(logging.getLevelNamesMapping().keys())),
)
@click.option(
    "--debug/--no-debug",
    default=True,
    help="automatically drop into ipdb on error",
)
@click.option(
    "--rich-traceback/--no-rich-traceback",
    default=True,
    help="use rich for traceback",
)
@click.option(
    "--echo-queries/--no-echo-queries",
    default=False,
    help="echo all sqlalchemy queries",
)
def cli(log_level, debug, rich_traceback, echo_queries):
    if debug or rich_traceback:

        def excepthook(type, value, tb):
            if issubclass(type, KeyboardInterrupt):
                sys.__excepthook__(type, value, tb)
                return
            if rich_traceback:
                traceback_console = Console(stderr=True)
                traceback_console.print(
                    Traceback.from_exception(type, value, tb),
                )
            else:
                traceback.print_tb(tb)
            if debug:
                ipdb.post_mortem(tb)

    if echo_queries:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

    sys.excepthook = excepthook
    LOGGER.setLevel(logging.getLevelName(log_level))


@cli.command(help="Open a pgcli shell to the postgres db")
def sql():
    _open_tunnel()
    import subprocess

    from cacafo.db.session import get_postgres_uri

    subprocess.run(["pgcli", get_postgres_uri()], check=True)


def _open_tunnel():
    import subprocess

    from rl.utils.io import getenv

    try:
        subprocess.run(
            f"curl localhost:{getenv('SSH_LOCAL_PORT')}".split(),
            check=True,
            timeout=1,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        match e.returncode:
            case 7:
                pass
            case 52:
                print(f"Tunnel is already running on port {getenv('SSH_LOCAL_PORT')}")
                return
            case _:
                raise ValueError(f"Tunnel is in weird state: {e}")

    subprocess.Popen(
        " ".join(
            [
                "nohup",
                "ssh",
                "-L",
                f"{getenv('SSH_LOCAL_PORT')}:localhost:{getenv('SSH_REMOTE_PORT')}",
                "-NT",
                getenv("SSH_ALIAS"),
            ]
        ),
        shell=True,
    )


@cli.command(help="Open a tunnel to the LCR postgres db")
def tunnel():
    _open_tunnel()


@cli.command(help="Open visidata view of the postgres db with the env settings")
def vd():
    _open_tunnel()
    import sys

    from cacafo.db.session import get_postgres_uri

    visidata_postgres_uri = get_postgres_uri().replace("postgresql", "postgres")
    binary = Path(sys.executable)
    binary = binary.parent / "vd"
    if not binary.exists():
        raise FileNotFoundError(
            f"Could not find visidata binary at {binary}; make sure project is installed with dev dependencies."
        )
    subprocess.run([str(binary), visidata_postgres_uri], check=True)


# ruff: noqa: F401, F841
@cli.command(help="Open an ipython shell with helpful project imports")
def shell():
    _open_tunnel()
    import hashlib
    from datetime import datetime
    from pathlib import Path

    import geoalchemy2 as ga
    import geopandas as gpd
    import more_itertools as mit
    import pyproj
    import rich.pretty
    import rl.utils.io
    import shapely as shp
    import shapely.ops
    import shapely.wkt as wkt
    import sqlalchemy as sa
    from geoalchemy2 import Geometry
    from geoalchemy2.shape import to_shape
    from rich import print
    from sqlalchemy.dialects.postgresql import JSON

    import cacafo.db.models as models
    import cacafo.query
    from cacafo.db.session import new_session
    from cacafo.transform import to_meters, to_wgs

    def vim(string, wrap=True):
        import os
        import subprocess
        import tempfile
        import textwrap as tw

        editor = os.environ.get("EDITOR", "vim")
        if wrap:
            string = tw.fill(string, width=80, replace_whitespace=False)
        with tempfile.NamedTemporaryFile(mode="w+") as tf:
            tf.write(string)
            tf.flush()
            subprocess.call([editor, tf.name])
            tf.seek(0)
            return tf.read()

    m = models
    s = new_session()
    session = s
    start_ipython(argv=[], user_ns=locals())


@cli.command(help="Print the location of any object in the database")
@click.argument("table_name")
@click.argument("id", type=int, required=False, default=None)
@click.option(
    "--column",
    type=(str, str),
    help="Column name and value to filter by",
    multiple=True,
)
def whereis(table_name, id, column):
    import rich
    from geoalchemy2.shape import to_shape
    from sqlalchemy import select

    from cacafo.db.models import get_model_by_table_name
    from cacafo.db.session import new_session

    session = new_session()
    model = get_model_by_table_name(table_name)

    stmt = select(model)

    if id is not None:
        stmt = stmt.where(model.id == id)
    elif column is not None:
        columns = column
        for column, value in columns:
            if hasattr(model, column):
                stmt = stmt.where(getattr(model, column) == value)
            else:
                raise ValueError(f"Column {column} not found on model {model}")
    else:
        rich.print("[yellow] No identifier specified, selecting arbitrary row[/yellow]")
        stmt = stmt.limit(1)

    results = session.execute(stmt).scalars().all()

    if len(results) != 1:
        raise ValueError("Expected exactly one result, found: {}".format(len(results)))

    obj = results[0]
    location_attrs = [
        "geometry",
        "location",
        "registered_location",
        "geocoded_address_location",
    ]
    for loc in location_attrs:
        if hasattr(obj, loc) and getattr(obj, loc) is not None:
            geometry = getattr(obj, loc)
            geometry = to_shape(geometry)
            lat, lon = geometry.centroid.y, geometry.centroid.x
            rich.print(f"{table_name} {obj.id} @ {lat}, {lon}")
            return
    raise ValueError(f"Identified {table_name} with id {id}, but no location found")


from cacafo.building_relationships import _cli as building_relationships_cli

# ruff: noqa: E402
from cacafo.check import _cli as check_cli
from cacafo.data.ingest import _cli as ingest_cli
from cacafo.data.join import _cli as join_cli
from cacafo.export import _cli as export_cli
from cacafo.facilities import _cli as facilities_cli
from cacafo.naip import _cli as naip_cli
from cacafo.paper import _cli as paper_cli
from cacafo.urban_mask import _cli as urban_mask_cli

cli.add_command(check_cli)
cli.add_command(export_cli)
cli.add_command(naip_cli)
cli.add_command(ingest_cli)
cli.add_command(join_cli)
cli.add_command(facilities_cli)
cli.add_command(building_relationships_cli)
cli.add_command(paper_cli)
cli.add_command(urban_mask_cli)

if __name__ == "__main__":
    cli()
