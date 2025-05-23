import csv
import datetime
import json
import os
import typing as t
from dataclasses import dataclass

import geoalchemy2 as ga
import geopandas as gpd
import rich.progress
import rich_click as click
import shapely as shp
import sqlalchemy as sa
from rich.traceback import install
from rl.utils.io import get_data_path
from sqlalchemy.dialects import postgresql

import cacafo.data.source
import cacafo.db.models as m
import cacafo.owner_name_matching
import cacafo.transform
from cacafo.db.session import new_session

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
        def wrapper(overwrite=False, add=False, file_path=None):
            session = new_session()
            _preflight(session, model, overwrite, add)
            previous_num = session.execute(
                sa.select(sa.func.count()).select_from(model)
            ).scalar()
            func(session, file_path)
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
def county_group(session, file_path=None):
    path = file_path or cacafo.data.source.get("county_groups.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        county_groups = []
        for line in rich.progress.track(reader, description="Ingesting county groups"):
            county_groups.append(m.CountyGroup(name=line["Group Name"]))
        session.add_all(county_groups)


@ingestor(m.County, depends_on=[m.CountyGroup])
def county(session, file_path=None):
    county_group_name_to_id = {
        name: id
        for name, id in session.execute(
            sa.select(m.CountyGroup.name, m.CountyGroup.id)
        ).all()
    }

    path = file_path or cacafo.data.source.get("county_groups.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        county_name_to_group_id = {
            line["County"]: county_group_name_to_id[line["Group Name"]]
            for line in reader
        }

    counties_path = file_path or cacafo.data.source.get("counties.csv")
    with open(counties_path) as f:
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


@ingestor(m.Parcel, depends_on=[m.County])
def parcel(session, file_path=None):
    county_name_to_id = {
        name: id
        for name, id in session.execute(sa.select(m.County.name, m.County.id)).all()
    }

    parcels: dict[tuple[str, str], m.Parcel] = {
        (p.county.name, p.number): p
        for p in session.execute(sa.select(m.Parcel)).scalars().all()
    }

    path = file_path or cacafo.data.source.get("parcels.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        for line in rich.progress.track(reader, description="Ingesting parcels"):
            data = json.loads(line["data"])
            if isinstance(data, str):
                # some data is double-quoted bc of json->csv
                data = json.loads(data)
            if (line["county_name"], line["numb"]) in parcels:
                continue
            parcel = m.Parcel(
                owner=line["owner"],
                address=line["address"],
                number=line["numb"],
                data=data,
                inferred_geometry=None,
                county_id=county_name_to_id[line["county_name"]],
            )
            parcels[(line["county_name"], line["numb"])] = parcel

    locations_path = file_path or cacafo.data.source.get("parcel_locations.csv")
    with open(locations_path) as f:
        reader = csv.DictReader(f)
        # each line lists a county,number,lat,lon
        for line in rich.progress.track(
            reader, description="Ingesting parcel locations"
        ):
            county, number, lat, lon = (
                line["county_name"],
                line["numb"],
                line["latitude"],
                line["longitude"],
            )
            if not county:
                # check every county
                possible_parcels = [
                    (c, number)
                    for c in county_name_to_id.keys()
                    if (c, number) in parcels
                ]
                if len(possible_parcels) == 0:
                    rich.print(
                        f"Skipping parcel num. {number} because it is not in any county"
                    )
                    continue
                if len(possible_parcels) != 1:
                    rich.print(
                        f"Skipping parcel {number} because it is in multiple counties: {possible_parcels}"
                    )
                    continue
                county = possible_parcels[0][0]
            if (county, number) not in parcels:
                continue
            if parcels[(county, number)].shp_inferred_geometry is not None:
                # merge the two geometries
                parcels[(county, number)].shp_inferred_geometry = parcels[
                    (county, number)
                ].shp_inferred_geometry.union(
                    shp.geometry.Point(float(lon), float(lat))
                )
            else:
                parcels[(county, number)].shp_inferred_geometry = shp.geometry.Point(
                    float(lon), float(lat)
                )
    for parcel in rich.progress.track(
        parcels.values(), description="Processing geometries"
    ):
        if not parcel.shp_inferred_geometry:
            continue
        # transform to meters, buffer, hull, transform back to latlon
        original_geometry = parcel.shp_inferred_geometry
        points_in_meters = cacafo.transform.to_meters(parcel.shp_inferred_geometry)
        buffer = points_in_meters.buffer(5)
        convex_hull = buffer.convex_hull
        parcel.shp_inferred_geometry = cacafo.transform.to_wgs(convex_hull)

        assert shp.geometry.shape(parcel.shp_inferred_geometry).is_valid
        assert shp.geometry.shape(parcel.shp_inferred_geometry).contains(
            original_geometry
        )

    session.add_all(parcels.values())


@ingestor(m.Permit)
def permit(session, file_path=None):
    geocoded_path = file_path or cacafo.data.source.get("geocoded_addresses.csv")
    with open(geocoded_path) as f:
        geocoded_addresses: dict[tuple[str, str], str] = {
            (line["Address"], line["City"]): (
                shp.geometry.Point(
                    float(line["Longitude"]), float(line["Latitude"])
                ).wkt,
            )
            for line in csv.DictReader(f)
        }
    permits_path = file_path or cacafo.data.source.get("permits.csv")
    with open(permits_path) as f:
        reader = csv.DictReader(f)
        permits = []
        for line in rich.progress.track(reader, description="Ingesting permits"):
            full_address = line["Facility Address"]
            try:
                tokens = [s.strip() for s in full_address.split(",")]
                address, city = tokens[:2]
            except ValueError:
                full_address = line["Agency Address"]
                tokens = [s.strip() for s in full_address.split(",")]
                address, city = tokens[:2]
            geom = None
            try:
                geom = shp.geometry.Point(
                    float(line["Longitude"]), float(line["Latitude"])
                ).wkt
            except ValueError:
                geom = None
                click.secho(
                    f"Permit {line['WDID']} has invalid coordinates",
                    fg="yellow",
                )
            permits.append(
                m.Permit(
                    data=line,
                    registered_location=geom,
                    geocoded_address_location=geocoded_addresses.get((address, city)),
                )
            )
        session.add_all(permits)


@ingestor(m.Image)
def image(session, file_path=None):
    path = file_path or cacafo.data.source.get("images.csv")
    with open(path) as f:
        county_name_to_id = {
            name: id
            for name, id in session.execute(sa.select(m.County.name, m.County.id)).all()
        }
        reader = list(csv.DictReader(f))
        images = []
        for line in rich.progress.track(reader, description="Ingesting images"):
            images.append(
                m.Image(
                    name=line["name"],
                    county_id=county_name_to_id[line["county"]],
                    geometry=shp.box(
                        float(line["lon_min"]),
                        float(line["lat_min"]),
                        float(line["lon_max"]),
                        float(line["lat_max"]),
                    ).wkt,
                    bucket=line["bucket"] or None,
                )
            )
            if len(images) > 50000:
                session.add_all(images)
                session.flush()
                images = []
        session.add_all(images)
    path = cacafo.data.source.get("post_hoc.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found, post hoc data required.")
    with open(path) as f:
        reader = csv.DictReader(f)
        names = {line["image_name"] for line in reader}
        # update label_reason to be post hoc for these images
        session.execute(
            sa.update(m.Image)
            .where(m.Image.name.in_(names))
            .values(label_reason="post_hoc")
        )


def _dig(d: t.Sequence, *keys: list[t.Any]) -> t.Any:
    if not keys:
        return d
    try:
        return _dig(d[keys[0]], *keys[1:])
    except (KeyError, IndexError, TypeError):
        return None


@ingestor(m.ImageAnnotation, depends_on=[m.Image])
def image_annotation(session, file_path=None):
    path = file_path or cacafo.data.source.get("annotations.jsonl")
    with open(path) as f:
        image_name_to_id = {
            name: id
            for name, id in session.execute(sa.select(m.Image.name, m.Image.id)).all()
        }
        annotations = []
        lines = [json.loads(line.strip()) for line in f.readlines() if line.strip()]
        for line in rich.progress.track(
            lines, description="Ingesting image annotations"
        ):
            filename = _dig(line, "filename") or _dig(line, "name")
            if filename is None:
                continue
            image = (image_name_to_id[filename.split("/")[-1].replace(".jpeg", "")],)
            annotations.append(
                {
                    "data": line,
                    "annotated_at": datetime.datetime.fromisoformat(
                        _dig(line, "annotations", 0, "createdAt") or line["createdAt"]
                    ),
                    "image_id": image[0],
                }
            )
            if len(annotations) > 5000:
                session.execute(
                    postgresql.insert(m.ImageAnnotation).on_conflict_do_nothing(
                        index_elements=[m.ImageAnnotation.hash]
                    ),
                    annotations,
                )
                annotations = []
        session.execute(
            postgresql.insert(m.ImageAnnotation).on_conflict_do_nothing(
                index_elements=[m.ImageAnnotation.hash]
            ),
            annotations,
        )


def _get_date(date: str, round_down: bool = False) -> t.Optional[datetime.datetime]:
    if date == "-" or not date:
        return None
    if round_down:
        return datetime.datetime.fromisoformat(f"{date}-01-01")
    return datetime.datetime.fromisoformat(f"{date}-12-31")


@ingestor(m.ConstructionAnnotation)
def construction_annotation(session, file_path=None):
    path = file_path or cacafo.data.source.get("construction_dating.csv")
    with open(path) as f:
        annotations = []
        lines = [line for line in csv.DictReader(f)]
        for line in rich.progress.track(
            lines, description="Ingesting construction annotations"
        ):
            if _get_date(line["construction_upper"]) is None:
                continue
            annotations.append(
                m.ConstructionAnnotation(
                    data=line,
                    location=shp.geometry.Point(
                        float(line["longitude"]), float(line["latitude"])
                    ).wkt,
                    construction_lower_bound=_get_date(
                        line["construction_lower"], round_down=True
                    ),
                    construction_upper_bound=_get_date(
                        line["construction_upper"],
                    ),
                    destruction_lower_bound=_get_date(
                        line["destruction_lower"], round_down=True
                    ),
                    destruction_upper_bound=_get_date(
                        line["destruction_upper"],
                    ),
                    significant_population_change=line[
                        "significant_animal_population_change"
                    ].lower()
                    == "true",
                    is_primarily_indoors="indoor" in line["where_animals_stay"],
                    has_lagoon=line["has_lagoon"].lower() == "true",
                    annotated_on=datetime.datetime.strptime(
                        line["processed_on"], "%m/%d/%Y"
                    ),
                )
            )
        session.add_all(annotations)


@ingestor(m.AnimalTypeAnnotation)
def animal_type_annotation(session, file_path=None):
    path = file_path or cacafo.data.source.get("animal_typing.csv")
    with open(path) as f:
        reader = list(csv.DictReader(f))
        animal_types = []
        for line in rich.progress.track(reader, description="Ingesting animal types"):
            for label in (
                line["type"],
                line["secondary_type"],
                line["subtype"],
            ):
                if not label:
                    continue
                animal_types.append(
                    m.AnimalTypeAnnotation(
                        animal_type=label,
                        location=shp.geometry.Point(
                            float(line["longitude"]), float(line["latitude"])
                        ).wkt,
                        annotated_on=datetime.datetime.fromisoformat(
                            line["annotated_before"]
                        ),
                        annotated_by=line["labeler"],
                        notes=line["notes"] or "",
                        annotation_facility_hash=line.get("annotation_facility_hash"),
                    )
                )
        session.add_all(animal_types)
        session.commit()


@ingestor(m.CafoAnnotation)
def cafo_annotation(session, file_path=None):
    if file_path:
        raise ValueError("file_path is not supported for cafo_annotation")
    # first ingest from animal typing, then from construction
    with open(cacafo.data.source.get("animal_typing.csv")) as f:
        animal_typing = list(csv.DictReader(f))
        cafo_annotations = []
        for line in rich.progress.track(
            animal_typing, description="Ingesting CAFO annotations from animal typing"
        ):
            if not line["is_cafo"] and not line["is_afo"]:
                continue
            is_afo = line["is_afo"].lower() == "true"
            is_cafo = line["is_cafo"].lower() == "true"
            # we used to consider feedlots not CAFOs
            # but now we do
            if is_afo and "feedlot" in line["notes"].lower():
                is_cafo = True
            cafo_annotations.append(
                m.CafoAnnotation(
                    location=shp.geometry.Point(
                        float(line["longitude"]), float(line["latitude"])
                    ).wkt,
                    annotated_on=datetime.datetime.fromisoformat(
                        line["annotated_before"]
                    ),
                    annotation_facility_hash=line.get("annotation_facility_hash")
                    or line.get("uuid"),
                    annotated_by=line["labeler"],
                    is_cafo=is_cafo,
                    is_afo=is_afo,
                    annotation_phase="animal typing",
                )
            )
        session.add_all(cafo_annotations)
    with open(cacafo.data.source.get("construction_dating.csv")) as f:
        construction_annotations = list(csv.DictReader(f))
        for line in rich.progress.track(
            construction_annotations,
            description="Ingesting CAFO annotations from construction dating",
        ):
            if not line["latitude"] or not line["longitude"]:
                continue
            cafo_annotations.append(
                m.CafoAnnotation(
                    location=shp.geometry.Point(
                        float(line["longitude"]), float(line["latitude"])
                    ).wkt,
                    annotated_on=datetime.datetime.strptime(
                        line["processed_on"], "%m/%d/%Y"
                    ),
                    annotation_facility_hash=line.get("uuid"),
                    is_cafo=line.get("is_cafo").lower() == "true",
                    is_afo=line.get("is_cafo").lower() == "true",
                    annotated_by=line["annotator"],
                    annotation_phase="construction dating",
                )
            )
        session.add_all(cafo_annotations)
        session.commit()


@ingestor(m.ParcelOwnerNameAnnotation)
def parcel_owner_name_annotation(session, file_path=None):
    path = file_path or cacafo.data.source.get("parcel_name_annotations.csv")
    with open(path) as f:
        reader = list(csv.DictReader(f))
        annotations = []
        for line in rich.progress.track(
            reader, description="Ingesting parcel owner relationship annotations"
        ):
            annotations.append(
                m.ParcelOwnerNameAnnotation(
                    owner_name=line["owner_1"],
                    related_owner_name=line["owner_2"],
                    matched=bool(int(line["create_override_match"])),
                    annotated_on=datetime.datetime.fromisoformat(line["annotated_on"]),
                    annotated_by=line["annotated_by"],
                )
            )
        session.add_all(annotations)
        session.commit()


@ingestor(m.Building, depends_on=[m.ImageAnnotation])
def building(session, file_path=None):
    from cacafo.dataloop import get_geometries_from_annnotation_data

    buildings = []
    name_to_image = {
        image.name: image
        for image in session.execute(sa.select(m.Image)).scalars().all()
    }
    annotations = session.execute(sa.select(m.ImageAnnotation)).scalars()
    for annotation in rich.progress.track(
        annotations, description="Ingesting buildings"
    ):
        data = annotation.data
        image_name = data["name"].split(".")[0].strip("/")
        image = name_to_image.get(image_name)
        geometries = get_geometries_from_annnotation_data(data)
        for geometry in geometries:
            image_latlon_poly = image.from_xy_to_lat_lon(geometry)
            buildings.append(
                {
                    "image_annotation_id": annotation.id,
                    "excluded_at": None,
                    "exclude_reason": None,
                    "parcel_id": None,
                    "image_xy_geometry": geometry.wkt,
                    "geometry": image_latlon_poly.wkt,
                }
            )
    session.execute(
        postgresql.insert(m.Building).on_conflict_do_nothing(
            index_elements=[m.Building.hash]
        ),
        buildings,
    )
    session.commit()
    from cluster.exclude_overlapping import exclude_overlapping_buildings

    exclude_overlapping_buildings(session)


@ingestor(m.UrbanMask)
def urban_mask(session, file_path=None):
    path = file_path or get_data_path("source/census/urban_mask_2019")
    df = gpd.read_file(path)
    df.crs = "EPSG:4326"
    for _, row in rich.progress.track(
        df.iterrows(), description="Ingesting urban mask"
    ):
        if ", CA" not in row["NAME10"]:
            continue
        session.add(
            m.UrbanMask(
                **(
                    {k.lower().strip("10"): row[k] for k in row.keys()}
                    | {"geometry": ga.WKTElement(row["geometry"].wkt, srid=4326)}
                )
            )
        )
    session.commit()


@ingestor(m.CensusBlock)
def census_block(session, file_path=None):
    path = file_path or get_data_path("source/census/block_2024")
    df = gpd.read_file(path)
    df.crs = "EPSG:4326"
    for _, row in rich.progress.track(
        df.iterrows(), description="Ingesting census blocks"
    ):
        geom = row["geometry"]
        if geom.geom_type == "Polygon":
            geom = shp.geometry.MultiPolygon([geom])
        geom = ga.WKTElement(geom.wkt, srid=4326)
        session.add(
            m.CensusBlock(
                **(
                    {k.lower().strip("20"): row[k] for k in row.keys()}
                    | {"geometry": geom}
                )
            )
        )
    session.commit()


@ingestor(m.IrrAnnotation, depends_on=[m.Image])
def irr_annotation(session, file_path=None):
    file_path = file_path or get_data_path("source/ca_irr")
    annotations = []
    for fn in os.listdir(file_path):
        if os.path.splitext(fn)[-1] != ".jsonl":
            continue
        path = os.path.join(file_path, fn)
        annotator = fn.split("_")[
            0
        ]  # first part of the file name should be annotator name
        with open(path) as f:
            image_name_to_id = {
                name: id
                for name, id in session.execute(
                    sa.select(m.Image.name, m.Image.id)
                ).all()
            }
            lines = [json.loads(line.strip()) for line in f.readlines() if line.strip()]
            for line in rich.progress.track(
                lines, description="Ingesting IRR image annotations"
            ):
                filename = _dig(line, "filename") or _dig(line, "name")
                if filename is None:
                    continue
                image = (
                    image_name_to_id[filename.split("/")[-1].replace(".jpeg", "")],
                )
                annotations.append(
                    {
                        "annotator": annotator,
                        "data": line,
                        "annotated_at": datetime.datetime.fromisoformat(
                            _dig(line, "annotations", 0, "createdAt")
                            or line["createdAt"]
                        ),
                        "image_id": image[0],
                    }
                )
        if len(annotations) > 5000:
            session.execute(
                postgresql.insert(m.IrrAnnotation).on_conflict_do_nothing(
                    index_elements=[m.IrrAnnotation.hash]
                ),
                annotations,
            )
            annotations = []
    session.execute(
        postgresql.insert(m.IrrAnnotation).on_conflict_do_nothing(
            index_elements=[m.IrrAnnotation.hash]
        ),
        annotations,
    )


def status():
    session = new_session()
    subclasses = m.Base.__subclasses__()
    for model in subclasses:
        if not _is_populated(session, model):
            click.secho(
                f"Table {model} is not populated",
                fg="red",
            )
        else:
            click.secho(
                f"Table {model} is populated",
                fg="green",
            )


@click.command("ingest", help="Ingest data into the database")
@click.option("--overwrite", is_flag=True)
@click.option("--add", is_flag=True)
@click.option("--file-path", help="Override default input file path", type=click.Path())
@click.argument(
    "tablename", type=click.Choice(list(Ingestor.instances.keys()) + ["status"])
)
def _cli(tablename, overwrite, add, file_path):
    if tablename == "status":
        status()
        return
    ingestor = Ingestor.instances[tablename]
    ingestor.func(overwrite=overwrite, add=add, file_path=file_path)
