import csv
import datetime
import json

import geopandas as gpd
import pandas as pd
import peewee as pw
import rich_click as click
import rl.utils.io
from tqdm import tqdm

from cacafo.db.models import *


def export_csv(output_path: str):
    facility_id_to_county = {
        row["id"]: County.geocode(lon=row["longitude"], lat=row["latitude"]).name
        for row in Facility.select(
            Facility.id, Facility.longitude, Facility.latitude
        ).dicts()
    }
    rows = (
        Facility.select(
            Facility.id.alias("facility_id"),
            Facility.uuid.alias("facility_uuid"),
            Facility.latitude,
            Facility.longitude,
            ConstructionAnnotation.id.alias("construction_annotation_id"),
            ConstructionAnnotation.construction_lower_bound.alias(
                "construction_lower_bound"
            ),
            ConstructionAnnotation.construction_upper_bound.alias(
                "construction_upper_bound"
            ),
            ConstructionAnnotation.destruction_lower_bound.alias(
                "destruction_lower_bound"
            ),
            ConstructionAnnotation.destruction_upper_bound.alias(
                "destruction_upper_bound"
            ),
            ConstructionAnnotation.significant_population_change.alias(
                "significant_population_change"
            ),
            ConstructionAnnotation.indoor_outdoor.alias("indoor_outdoor"),
            ConstructionAnnotation.has_lagoon.alias("has_lagoon"),
        )
        .join(ConstructionAnnotation)
        .dicts()
    )
    rows = list(rows)
    rows = [{"county": facility_id_to_county[r["facility_id"]], **r} for r in rows]
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    writer = csv.DictWriter(
        open(output_path, "w"),
        fieldnames=rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(rows)
    return rows


def export_geojson(output_path: str):
    query = pw.prefetch(
        Facility.select(),
        Permit.select(),
        Building.select(),
        Parcel.select(),
        County.select(County.id, County.name),
        FacilityPermittedLocation.select(),
        PermittedLocation.select(),
        FacilityAnimalType.select(),
        AnimalType.select(),
        ConstructionAnnotation.select(),
    )
    features = [facility.to_geojson_feature() for facility in tqdm(query)]
    return {
        "type": "FeatureCollection",
        "features": features,
    }
    with open(output_path, "w") as f:
        json.dump(geojson, f)


@click.command()
@click.option(
    "--output-path",
    "-o",
    default=str(
        rl.utils.io.get_data_path()
        / f"{datetime.datetime.now().strftime('%Y-%m-%d')}.geojson"
    ),
    help="Output path",
)
@click.option(
    "--format",
    "-f",
    "format_",
    default=None,
    help="Output format; inferred from output path if not provided",
)
def export(output_path: str, format_: str):
    if format_ is None:
        format_ = output_path.split(".")[-1]
    match format_:
        case "geojson":
            data = export_geojson(output_path)
        case "csv":
            data = export_csv(output_path)
        case _:
            raise ValueError(f"Unknown format: {format}")
    click.echo(f"Exported data to {output_path}")
