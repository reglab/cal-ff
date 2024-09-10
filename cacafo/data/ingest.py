import csv
import typing as t
from dataclasses import dataclass

import pyproj
import rich.progress
import rich_click as click
import shapely as shp
import sqlalchemy as sa
from rich.traceback import install

import cacafo.data.source
import cacafo.db.sa_models as m
from cacafo.db.session import get_sqlalchemy_session

install(show_locals=True)


DEFAULT_EPSG = 4326
CA_EPSG = 3311


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
    to_meters = pyproj.Transformer.from_crs(DEFAULT_EPSG, CA_EPSG, always_xy=True)
    to_latlon = pyproj.Transformer.from_crs(CA_EPSG, DEFAULT_EPSG, always_xy=True)
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


@click.command("ingest", help="Ingest data into the database")
@click.option("--overwrite", is_flag=True)
@click.option("--add", is_flag=True)
@click.argument("tablename", type=click.Choice(Ingestor.instances.keys()))
def _cli(tablename, overwrite, add):
    ingestor = Ingestor.instances[tablename]
    ingestor.func(overwrite=overwrite, add=add)
