import logging
import sys
import traceback

import ipdb
import rich_click as click
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
    import os
    import subprocess

    from cacafo.db.session import get_postgres_uri

    subprocess.run(["pgcli", get_postgres_uri()], check=True)


@cli.command()
def tunnel():
    import os
    import subprocess

    import requests
    from rl.utils.io import getenv

    try:
        proc = subprocess.run(
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


from cacafo.check import check
from cacafo.export import export
from cacafo.paper import cmd_generate

cli.add_command(check)
cli.add_command(export)
cli.add_command(cmd_generate)

if __name__ == "__main__":
    cli()
