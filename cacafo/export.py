import csv
import datetime
import json

import rich_click as click
import rl.utils.io
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from cacafo.db.sa_models import ConstructionAnnotation, County, Facility


def export_csv(session: Session, output_path: str):
    facility_id_to_county = {
        row.id: County.geocode(session, lon=row.longitude, lat=row.latitude).name
        for row in session.execute(
            select(
                Facility.id,
                Facility.geometry.ST_X().label("longitude"),
                Facility.geometry.ST_Y().label("latitude"),
            )
        ).all()
    }

    rows = session.execute(
        select(
            Facility.id.label("facility_id"),
            Facility.hash.label("facility_uuid"),
            Facility.geometry.ST_Centroid().ST_X().label("longitude"),
            Facility.geometry.ST_Centroid().ST_Y().label("latitude"),
            ConstructionAnnotation.id.label("construction_annotation_id"),
            ConstructionAnnotation.construction_lower_bound,
            ConstructionAnnotation.construction_upper_bound,
            ConstructionAnnotation.destruction_lower_bound,
            ConstructionAnnotation.destruction_upper_bound,
            ConstructionAnnotation.significant_population_change,
            ConstructionAnnotation.is_primarily_indoors.label("indoor_outdoor"),
            ConstructionAnnotation.has_lagoon,
        )
        .join(ConstructionAnnotation, Facility.id == ConstructionAnnotation.facility_id)
        .where(Facility.archived_at.is_(None))
    ).all()

    rows = [row._asdict() for row in rows]
    rows = [{"county": facility_id_to_county[r["facility_id"]], **r} for r in rows]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export_geojson(session: Session, output_path: str):
    query = select(Facility).where(Facility.archived_at.is_(None))
    facilities = session.execute(query).scalars().all()
    features = [facility.to_geojson_feature() for facility in tqdm(facilities)]
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    return geojson


@click.command("export")
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
def _cli(output_path: str, format_: str | None):
    from cacafo.db.session import get_sqlalchemy_session

    if format_ is None:
        format_ = output_path.split(".")[-1]

    with get_sqlalchemy_session() as session:
        match format_:
            case "geojson":
                export_geojson(session, output_path)
            case "csv":
                export_csv(session, output_path)
            case _:
                raise ValueError(f"Unknown format: {format_}")

    click.echo(f"Exported data to {output_path}")
