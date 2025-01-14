import csv
import datetime
import itertools
import json
import random

import geoalchemy2 as ga
import geopandas as gpd
import rich_click as click
import rl.utils.io
import sqlalchemy as sa
from sqlalchemy.orm import Session
from tqdm import tqdm

import cacafo.db.models as m
import cacafo.query

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
    facilities = (
        session.execute(
            cacafo.query.cafos().options(
                sa.orm.joinedload(m.Facility.all_construction_annotations),
                sa.orm.joinedload(m.Facility.all_animal_type_annotations),
                sa.orm.joinedload(m.Facility.best_permits),
                sa.orm.joinedload(m.Facility.county),
            )
        )
        .unique()
        .scalars()
        .all()
    )
    rows = [
        {
            "facility_id": facility.id,
            "facility_hash": facility.hash,
            "latitude": facility.shp_geometry.centroid.y,
            "longitude": facility.shp_geometry.centroid.x,
            "construction_lower_bound": facility.construction_annotation.construction_lower_bound,
            "construction_upper_bound": facility.construction_annotation.construction_upper_bound,
            "destruction_lower_bound": facility.construction_annotation.destruction_lower_bound,
            "destruction_upper_bound": facility.construction_annotation.destruction_upper_bound,
            "significant_population_change": facility.construction_annotation.significant_population_change,
            "indoor_outdoor": facility.construction_annotation.is_primarily_indoors,
            "has_lagoon": facility.construction_annotation.has_lagoon,
            "animal_type": facility.animal_type_str,
        }
        for facility in tqdm(facilities)
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


@exporter("facilities", "geojson")
def facilities_geojson(session: Session, output_path: str):
    facilities = (
        session.execute(
            cacafo.query.cafos().options(
                sa.orm.selectinload(m.Facility.all_construction_annotations),
                sa.orm.selectinload(m.Facility.all_animal_type_annotations),
                # sa.orm.selectinload(m.Facility.best_permits),
                # sa.orm.joinedload(m.Facility.county),
                sa.orm.selectinload(m.Facility.all_buildings, m.Building.parcel),
            )
        )
        .unique()
        .scalars()
        .all()
    )
    features = [facility.to_geojson_feature() for facility in tqdm(facilities)]
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    return features


@exporter("buildings", "geojson")
def export_buildings_geojson(session: Session, output_path: str):
    query = sa.select(m.Building).options(sa.orm.joinedload(m.Building.parcel))
    buildings = session.execute(query).scalars().all()
    features = [building.to_geojson_feature() for building in tqdm(buildings)]
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


@exporter("urban_mask", "geojson")
def export_urban_mask_geojson(session: Session, output_path: str):
    query = sa.select(m.UrbanMask)
    urban_masks = session.execute(query).scalars().all()
    df = gpd.GeoDataFrame(
        {
            column.key: [getattr(urban_mask, column.key) for urban_mask in urban_masks]
            for column in sa.inspect(m.UrbanMask).columns
        }
        | {
            "geometry": [
                ga.shape.to_shape(urban_mask.geometry) for urban_mask in urban_masks
            ],
        }
    )
    text = df.to_json()
    with open(output_path, "w") as f:
        f.write(text)
    return df


@exporter("construction_annotations", "csv")
def construction_annotations_csv(session: Session, output_path: str):
    rows = session.execute(
        sa.select(
            m.ConstructionAnnotation.id.label("construction_annotation_id"),
            m.ConstructionAnnotation.facility_id,
            m.ConstructionAnnotation.construction_lower_bound,
            m.ConstructionAnnotation.construction_upper_bound,
            m.ConstructionAnnotation.destruction_lower_bound,
            m.ConstructionAnnotation.destruction_upper_bound,
            m.ConstructionAnnotation.significant_population_change,
            m.ConstructionAnnotation.is_primarily_indoors.label("indoor_outdoor"),
            m.ConstructionAnnotation.has_lagoon,
            sa.cast(
                m.ConstructionAnnotation.location,
                ga.Geometry,
            )
            .ST_X()
            .label("longitude"),
            sa.cast(
                m.ConstructionAnnotation.location,
                ga.Geometry,
            )
            .ST_Y()
            .label("latitude"),
        )
    ).all()
    rows = [row._asdict() for row in rows]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


@exporter("animal_type_annotations", "csv")
def animal_type_annotations_csv(session: Session, output_path: str):
    rows = session.execute(
        sa.select(
            m.AnimalTypeAnnotation.id.label("animal_type_annotation_id"),
            m.AnimalTypeAnnotation.facility_id,
            m.AnimalTypeAnnotation.animal_type,
            sa.cast(
                m.AnimalTypeAnnotation.location,
                ga.Geometry,
            )
            .ST_X()
            .label("longitude"),
            sa.cast(
                m.AnimalTypeAnnotation.location,
                ga.Geometry,
            )
            .ST_Y()
            .label("latitude"),
        )
    ).all()
    rows = [row._asdict() for row in rows]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


@exporter("animal_typing_sheet", "csv")
def animal_typing_sheet(session: Session, output_path: str):
    subq = cacafo.query.cafos().subquery()
    rows = (
        session.execute(
            sa.select(
                m.Facility,
            )
            .join(subq, m.Facility.id == subq.c.id)
            .options(
                sa.orm.joinedload(m.Facility.all_animal_type_annotations),
                sa.orm.joinedload(m.Facility.all_cafo_annotations),
                sa.orm.joinedload(m.Facility.best_permits),
            )
            .where(m.Facility.archived_at.is_(None))
        )
        .unique()
        .scalars()
        .all()
    )
    places_to_annotate = []
    for facility in rows:
        if not facility.animal_types:
            places_to_annotate.append(
                {
                    "annotated_before": "",
                    "annotation_facility_hash": facility.hash,
                    "latitude": facility.shp_geometry.centroid.y,
                    "longitude": facility.shp_geometry.centroid.x,
                    "labeler": "",
                    "is_afo": "",
                    "is_cafo": "",
                    "type": "",
                    "secondary_type": "",
                    "subtype": "",
                    "notes": "",
                }
            )

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=places_to_annotate[0].keys())
        writer.writeheader()
        writer.writerows(places_to_annotate)
    return rows


@exporter("construction_dating_sheet", "csv")
def construction_dating_sheet(session: Session, output_path: str):
    rows = (
        session.execute(
            sa.select(
                m.Facility,
            )
            .options(
                sa.orm.joinedload(m.Facility.all_animal_type_annotations),
                sa.orm.joinedload(m.Facility.all_cafo_annotations),
                sa.orm.joinedload(m.Facility.best_permits),
                sa.orm.joinedload(m.Facility.all_construction_annotations),
            )
            .where(m.Facility.archived_at.is_(None))
        )
        .unique()
        .scalars()
        .all()
    )
    places_to_annotate = []
    for facility in rows:
        if facility.is_cafo and not facility.construction_annotation:
            places_to_annotate.append(
                {
                    "processed_on": "",
                    "annotator": "",
                    "cafo_id": facility.id,
                    "cafo_uuid": facility.hash,
                    "latitude": ga.shape.to_shape(facility.geometry).centroid.y,
                    "longitude": ga.shape.to_shape(facility.geometry).centroid.x,
                    "is_cafo": "",
                    "centered_on_facility": "",
                    "construction_lower": "",
                    "construction_upper": "",
                    "destruction_lower": "",
                    "destruction_upper": "",
                    "significant_population_change": "",
                    "where_animals_stay": "",
                    "has_lagoon": "",
                    "cf_notes": "",
                    "reviewed_on": "",
                    "reviewed_by": "",
                    "reviewed_by_cf": "",
                    "cf_review_feedback": "",
                    "cf_correction_done": "",
                    "reglab_feedback": "",
                    "reannotate": "",
                }
            )

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=places_to_annotate[0].keys())
        writer.writeheader()
        writer.writerows(places_to_annotate)
    return places_to_annotate


@exporter("cafo_annotations", "csv")
def cafo_annotations_csv(session: Session, output_path: str):
    rows = session.execute(
        sa.select(
            m.CafoAnnotation.id.label("cafo_annotation_id"),
            m.CafoAnnotation.facility_id,
            m.CafoAnnotation.is_cafo,
            m.CafoAnnotation.is_afo,
            m.CafoAnnotation.annotated_on,
            m.CafoAnnotation.annotated_by,
            sa.cast(
                m.CafoAnnotation.location,
                ga.Geometry,
            )
            .ST_X()
            .label("longitude"),
            sa.cast(
                m.CafoAnnotation.location,
                ga.Geometry,
            )
            .ST_Y()
            .label("latitude"),
        )
    ).all()
    rows = [row._asdict() for row in rows]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


@exporter("irr_batch", "csv")
def irr_batch(session: Session, output_path: str):
    positive_images = session.execute(cacafo.query.positive_images()).scalars().all()
    labeled_negative_images = (
        session.execute(cacafo.query.labeled_negative_images()).scalars().all()
    )
    high_confidence_negative_images = (
        session.execute(cacafo.query.high_confidence_negative_images()).scalars().all()
    )
    low_confidence_negative_images = (
        session.execute(cacafo.query.low_confidence_negative_images()).scalars().all()
    )

    # assert that the classes are mutually exclusive
    positive_image_ids = {image.id for image in positive_images}
    labeled_negative_image_ids = {image.id for image in labeled_negative_images}
    high_confidence_negative_image_ids = {
        image.id for image in high_confidence_negative_images
    }
    low_confidence_negative_image_ids = {
        image.id for image in low_confidence_negative_images
    }
    for set_one, set_two in itertools.combinations(
        [
            positive_image_ids,
            labeled_negative_image_ids,
            high_confidence_negative_image_ids,
            low_confidence_negative_image_ids,
        ],
        2,
    ):
        assert not set_one & set_two
    # sample 50 random images from each class, and write their names to file
    writer = csv.DictWriter(
        open(output_path, "w", newline=""),
        fieldnames=["stratum", "image_name", "labeler"],
    )
    labeler_names = ["daniel", "nyambe", "jackline", "james"]
    rows = []
    writer.writeheader()
    for image_set, stratum in zip(
        [
            positive_images,
            labeled_negative_images,
            high_confidence_negative_images,
            low_confidence_negative_images,
        ],
        [
            "positive",
            "labeled_negative",
            "high_confidence_negative",
            "low_confidence_negative",
        ],
    ):
        random.shuffle(image_set)
        sample = image_set[:400]
        for images in zip(sample[::2], sample[1::2]):
            labelers_one = random.sample(labeler_names, 2)
            labelers_two = [a for a in labeler_names if a not in labelers_one]
            for im, adj in zip(images, (labelers_one, labelers_two)):
                for a in adj:
                    rows.append(
                        {
                            "stratum": stratum,
                            "image_name": im.name,
                            "labeler": a,
                        }
                    )
                    writer.writerow(rows[-1])
    return rows


@exporter("permits", "csv")
def permits_csv(session: Session, output_path: str):
    rows = session.execute(sa.select(m.Permit)).scalars().all()
    rows = [
        row.data
        | {
            "geocoded_address_latitude": (
                row.shp_geocoded_address_location
                and row.shp_geocoded_address_location.y
            ),
            "geocoded_address_longitude": (
                row.shp_geocoded_address_location
                and row.shp_geocoded_address_location.x
            ),
        }
        for row in rows
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


@click.command("export", help="Export data")
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
    from cacafo.db.session import new_session

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

    with new_session() as session:
        exporter = EXPORTERS[entity][format_]
        result = exporter(session, output_path)
        click.echo(f"Exported {len(result)} records to {output_path}")
