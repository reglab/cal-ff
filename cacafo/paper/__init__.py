import rich_click as click

from cacafo.paper.constants import _cli as constants_cli
from cacafo.paper.plot import _cli as plot_cli


@click.group("paper", help="Commands to generate figures and constants for the paper.")
def _cli():
    pass


_cli.add_command(constants_cli)
_cli.add_command(plot_cli)
