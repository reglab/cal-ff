import csv
import datetime
import json

import geoalchemy2 as ga
import geopandas as gpd
import rich_click as click
import rl.utils.io
import sqlalchemy as sa
from sqlalchemy.orm import Session
from tqdm import tqdm

import cacafo.db.sa_models as m

EXPORTERS = {}


def exporter(entity: str, format_: str):
    def decorator(func):
        func.entity = entity
        func.format = format_
        EXPORTERS[entity] = EXPORTERS.get(entity, {}) | {format_: func}
        return func

    return decorator


@exporter("facilities", "csv")
def facilities_csv(session: Session, output_path: str):
    facility_id_to_county = {
        row.id: m.County.geocode(session, lon=row.longitude, lat=row.latitude).name
        for row in session.execute(
            sa.select(
                m.Facility.id,
                m.Facility.geometry.ST_X().label("longitude"),
                m.Facility.geometry.ST_Y().label("latitude"),
            )
        ).all()
    }

    rows = session.execute(
        sa.select(
            m.Facility.id.label("facility_id"),
            m.Facility.hash.label("facility_uuid"),
            m.Facility.geometry.ST_Centroid().ST_X().label("longitude"),
            m.Facility.geometry.ST_Centroid().ST_Y().label("latitude"),
            m.ConstructionAnnotation.id.label("construction_annotation_id"),
            m.ConstructionAnnotation.construction_lower_bound,
            m.ConstructionAnnotation.construction_upper_bound,
            m.ConstructionAnnotation.destruction_lower_bound,
            m.ConstructionAnnotation.destruction_upper_bound,
            m.ConstructionAnnotation.significant_population_change,
            m.ConstructionAnnotation.is_primarily_indoors.label("indoor_outdoor"),
            m.ConstructionAnnotation.has_lagoon,
        )
        .join(
            m.ConstructionAnnotation,
            m.Facility.id == m.ConstructionAnnotation.facility_id,
        )
        .where(m.Facility.archived_at.is_(None))
    ).all()

    rows = [row._asdict() for row in rows]
    rows = [{"county": facility_id_to_county[r["facility_id"]], **r} for r in rows]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


@exporter("facilities", "geojson")
def export_geojson(session: Session, output_path: str):
    query = sa.select(m.Facility).where(m.Facility.archived_at.is_(None))
    facilities = session.execute(query).scalars().all()
    features = [facility.to_geojson_feature() for facility in tqdm(facilities)]
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    return features


@exporter("parcels", "geojson")
def export_parcels_geojson(session: Session, output_path: str):
    query = sa.select(m.Parcel).options(
        sa.orm.joinedload(m.Parcel.county, innerjoin=True)
    )
    parcels = session.execute(query).scalars().all()
    df = gpd.GeoDataFrame(
        {
            "id": [parcel.id for parcel in parcels],
            "geometry": [
                parcel.inferred_geometry and ga.shape.to_shape(parcel.inferred_geometry)
                for parcel in parcels
            ],
            "number": [parcel.number for parcel in parcels],
            "county": [parcel.county.name for parcel in parcels],
        }
    )
    text = df.to_json()
    with open(output_path, "w") as f:
        f.write(text)
    return df


@click.command("export")
@click.option(
    "--output-path",
    "-o",
    default="",
    help="Output path",
)
@click.option(
    "--format",
    "-f",
    "format_",
    default="",
    help="Output format; inferred from output path if not provided",
)
@click.argument("entity", type=click.Choice(EXPORTERS.keys()))
def _cli(entity, output_path, format_):
    from cacafo.db.session import get_sqlalchemy_session

    if not format_:
        if not output_path:
            # check if there's only one format for the entity
            formats = EXPORTERS[entity].keys()
            if len(formats) == 1:
                format_ = next(iter(formats))
            else:
                raise click.UsageError(
                    f"Output format must be provided when output path is not provided. Choices are: {', '.join(formats)}"
                )
        else:
            format_ = output_path.suffix[1:]

    if not output_path:
        output_path = (
            rl.utils.io.get_data_path()
            / f"{entity}_{datetime.datetime.now().strftime('%Y-%m-%d')}.{format_}"
        )

    with get_sqlalchemy_session() as session:
        exporter = EXPORTERS[entity][format_]
        result = exporter(session, output_path)
        click.echo(f"Exported {len(result)} records to {output_path}")
