import csv
import datetime
import json
import typing as t
from dataclasses import dataclass

import pyproj
import rich.progress
import rich_click as click
import shapely as shp
import sqlalchemy as sa
from rich.traceback import install
from sqlalchemy.dialects import postgresql

import cacafo.data.source
import cacafo.db.sa_models as m
import cacafo.owner_name_matching
from cacafo.constants import CA_SRID, DEFAULT_SRID
from cacafo.db.session import get_sqlalchemy_session

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
        def wrapper(overwrite=False, add=False):
            session = get_sqlalchemy_session()
            _preflight(session, model, overwrite, add)
            previous_num = session.execute(
                sa.select(sa.func.count()).select_from(model)
            ).scalar()
            func(session)
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
def county_group(session):
    with open(cacafo.data.source.get("county_groups.csv")) as f:
        reader = csv.DictReader(f)
        county_groups = []
        for line in rich.progress.track(reader, description="Ingesting county groups"):
            county_groups.append(m.CountyGroup(name=line["Group Name"]))
        session.add_all(county_groups)


@ingestor(m.County, depends_on=[m.CountyGroup])
def county(session):
    county_group_name_to_id = {
        name: id
        for name, id in session.execute(
            sa.select(m.CountyGroup.name, m.CountyGroup.id)
        ).all()
    }

    with open(cacafo.data.source.get("county_groups.csv")) as f:
        reader = csv.DictReader(f)
        county_name_to_group_id = {
            line["County"]: county_group_name_to_id[line["Group Name"]]
            for line in reader
        }

    with open(cacafo.data.source.get("counties.csv")) as f:
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
def parcel(session):
    county_name_to_id = {
        name: id
        for name, id in session.execute(sa.select(m.County.name, m.County.id)).all()
    }

    parcels: dict[tuple[str, str], m.Parcel] = {}

    with open(cacafo.data.source.get("parcels.csv")) as f:
        reader = csv.DictReader(f)
        for line in rich.progress.track(reader, description="Ingesting parcels"):
            parcel = m.Parcel(
                owner=line["owner"],
                address=line["address"],
                number=line["numb"],
                data=line,
                inferred_geometry=None,
                county_id=county_name_to_id[line["county_name"]],
            )
            parcels[(line["county_name"], line["numb"])] = parcel

    with open(cacafo.data.source.get("parcel_locations.csv")) as f:
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
            if parcels[(county, number)].inferred_geometry is not None:
                # merge the two geometries
                parcels[(county, number)].inferred_geometry = parcels[
                    (county, number)
                ].inferred_geometry.union(shp.geometry.Point(float(lon), float(lat)))
            else:
                parcels[(county, number)].inferred_geometry = shp.geometry.Point(
                    float(lon), float(lat)
                )
    to_meters = pyproj.Transformer.from_crs(DEFAULT_SRID, CA_SRID, always_xy=True)
    to_latlon = pyproj.Transformer.from_crs(CA_SRID, DEFAULT_SRID, always_xy=True)
    for parcel in rich.progress.track(
        parcels.values(), description="Processing geometries"
    ):
        if not parcel.inferred_geometry:
            continue
        # transform to meters, buffer, hull, transform back to latlon
        original_geometry = parcel.inferred_geometry
        points_in_meters = shp.ops.transform(
            to_meters.transform, parcel.inferred_geometry
        )
        buffer = points_in_meters.buffer(5)
        convex_hull = buffer.convex_hull
        parcel.inferred_geometry = shp.ops.transform(to_latlon.transform, convex_hull)

        assert shp.geometry.shape(parcel.inferred_geometry).is_valid
        assert shp.geometry.shape(parcel.inferred_geometry).contains(original_geometry)

        parcel.inferred_geometry = parcel.inferred_geometry.wkt

    session.add_all(parcels.values())


@ingestor(m.Permit)
def permit(session):
    with open(cacafo.data.source.get("geocoded_addresses.csv")) as f:
        geocoded_addresses: dict[tuple[str, str], str] = {
            (line["Address"], line["City"]): (
                shp.geometry.Point(
                    float(line["Longitude"]), float(line["Latitude"])
                ).wkt,
            )
            for line in csv.DictReader(f)
        }
    with open(cacafo.data.source.get("permits.csv")) as f:
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
def image(session):
    with open(cacafo.data.source.get("images.csv")) as f:
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


def _dig(d: t.Sequence, *keys: list[t.Any]) -> t.Any:
    if not keys:
        return d
    try:
        return _dig(d[keys[0]], *keys[1:])
    except (KeyError, IndexError, TypeError):
        return None


@ingestor(m.ImageAnnotation, depends_on=[m.Image])
def image_annotation(session):
    with open(cacafo.data.source.get("annotations.jsonl")) as f:
        image_name_to_id = {
            name: id
            for name, id in session.execute(sa.select(m.Image.name, m.Image.id)).all()
        }
        annotations = []
        lines = [json.loads(line) for line in f.readlines() if line.strip()]
        for line in rich.progress.track(
            lines, description="Ingesting image annotations"
        ):
            filename = _dig(line, "filename")
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
def construction_annotation(session):
    with open(cacafo.data.source.get("construction_dating.csv")) as f:
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
                    construction_lower_bound=_get_date(line["construction_lower"]),
                    construction_upper_bound=_get_date(
                        line["construction_upper"], round_down=True
                    ),
                    destruction_lower_bound=_get_date(line["destruction_lower"]),
                    destruction_upper_bound=_get_date(
                        line["destruction_upper"], round_down=True
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
def animal_type_annotation(session):
    with open(cacafo.data.source.get("animal_typing.csv")) as f:
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
                        notes=line["notes"],
                    )
                )
        session.add_all(animal_types)
        session.commit()


@ingestor(m.CafoAnnotation)
def cafo_annotation(session):
    # first ingest from animal typing, then from construction
    with open(cacafo.data.source.get("animal_typing.csv")) as f:
        animal_typing = list(csv.DictReader(f))
        cafo_annotations = []
        for line in rich.progress.track(
            animal_typing, description="Ingesting CAFO annotations from animal typing"
        ):
            if not line["is_cafo"] and not line["is_afo"]:
                continue
            is_afo = "n" not in line["is_afo"].lower()
            is_cafo = is_afo and ("n" not in line["is_cafo"].lower())
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
                    annotated_by=line["labeler"],
                    is_cafo=is_cafo,
                    is_afo=is_afo,
                )
            )
        session.add_all(cafo_annotations)
    with open(cacafo.data.source.get("construction_dating.csv")) as f:
        construction_annotations = list(csv.DictReader(f))
        for line in rich.progress.track(
            construction_annotations,
            description="Ingesting CAFO annotations from construction dating",
        ):
            cafo_annotations.append(
                m.CafoAnnotation(
                    location=shp.geometry.Point(
                        float(line["longitude"]), float(line["latitude"])
                    ).wkt,
                    annotated_on=datetime.datetime.strptime(
                        line["processed_on"], "%m/%d/%Y"
                    ),
                    is_cafo=True,
                    is_afo=True,
                    annotated_by=line["annotator"],
                )
            )
        session.add_all(cafo_annotations)
        session.commit()


@ingestor(m.ParcelOwnerNameAnnotation)
def parcel_owner_name_annotation(session):
    with open(cacafo.data.source.get("parcel_name_annotations.csv")) as f:
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
def building(session):
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


def status():
    session = get_sqlalchemy_session()
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
@click.argument(
    "tablename", type=click.Choice(list(Ingestor.instances.keys()) + ["status"])
)
def _cli(tablename, overwrite, add):
    if tablename == "status":
        status()
        return
    ingestor = Ingestor.instances[tablename]
    ingestor.func(overwrite=overwrite, add=add)
