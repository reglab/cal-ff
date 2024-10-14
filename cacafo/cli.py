import logging
import subprocess
import sys
import traceback

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


@cli.command()
def sql():
    import subprocess

    from cacafo.db.session import get_postgres_uri

    subprocess.run(["pgcli", get_postgres_uri()], check=True)


@cli.command()
def tunnel():
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


@cli.command()
def vd():
    from cacafo.db.session import get_postgres_uri

    visidata_postgres_uri = get_postgres_uri().replace("postgresql", "postgres")
    subprocess.run(["vd", visidata_postgres_uri])


# ruff: noqa: F401, F841
@cli.command()
def shell():
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

    import cacafo.db.sa_models as models
    from cacafo.db.session import get_sqlalchemy_session as get_session
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
    s = get_session()
    session = s
    start_ipython(argv=[], user_ns=locals())


from cacafo.building_relationships import _cli as building_relationships_cli

# ruff: noqa: E402
from cacafo.check import _cli as check_cli
from cacafo.data.ingest import _cli as ingest_cli
from cacafo.data.join import _cli as join_cli
from cacafo.export import _cli as export_cli
from cacafo.facilities import _cli as facilities_cli
from cacafo.naip import _cli as naip_cli
from cacafo.paper.constants import _cli as constant_cli
from cacafo.reports import _cli as reports_cli
from cacafo.vis import _cli as vis_cli

cli.add_command(check_cli)
cli.add_command(export_cli)
cli.add_command(naip_cli)
cli.add_command(reports_cli)
cli.add_command(ingest_cli)
cli.add_command(join_cli)
cli.add_command(facilities_cli)
cli.add_command(building_relationships_cli)
cli.add_command(vis_cli)
cli.add_command(constant_cli)

if __name__ == "__main__":
    cli()
