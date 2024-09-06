import json
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
import peewee as pw
import rich_click as click

import cacafo.db.models as m


@dataclass
class Report:
    name: str
    description: str
    func: callable

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


reports = {}


def report(description):
    def decorator(func):
        name = func.__name__.replace("_", "-")
        reports[name] = Report(name=name, description=description, func=func)
        return func

    return decorator


@report(
    "List all permits >200 animal counts that do not have any facility associated with them."
)
def unmatched_cafo_permits():
    unmatched_permits = (
        m.Permit.select()
        .where(
            ~m.Permit.id.in_(
                m.Permit.select(m.Permit.id)
                .join(m.PermitPermittedLocation)
                .join(m.PermittedLocation)
                .join(m.FacilityPermittedLocation)
            )
        )
        .dicts()
    )
    return pd.DataFrame(
        [
            p | {"location": f"{p['data']['Latitude']},{p['data']['Longitude']}"}
            for p in unmatched_permits
            if float(p["data"]["Cafo Population"] or "0.0") > 200
        ]
    )


@report("Geojson file of all buildings, whether CAFO or not.")
def all_buildings_geojson():
    buildings = m.Building.select()
    query = pw.prefetch(buildings, m.Facility)
    query = pw.prefetch(buildings, m.Parcel)
    query = pw.prefetch(buildings, m.Image)
    building = []
    for b in query:
        building.append(b.to_geojson_feature())
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": building,
        }
    )


@click.command(
    "reports",
    help="Run a report. Use list to see available reports, and report --help to see report-specific options.",
)
@click.argument(
    "report",
    type=click.Choice(reports.keys()),
    required=True,
)
@click.option("--output", type=click.Path(), help="Output file", default=None)
@click.option("--help", is_flag=True, help="Show help for the report")
def _cli(report, output, help):
    if help:
        print(f"{reports[report].name}: {reports[report].description}")
        return
    report = reports[report]
    result = report()
    if output:
        if isinstance(result, pd.DataFrame):
            with open(output, "w") as f:
                result.to_csv(f, index=False)
        else:
            with open(output, "w") as f:
                f.write(result)
    else:
        if isinstance(result, pd.DataFrame):
            print(result.to_csv(index=False))
        else:
            print(result)
