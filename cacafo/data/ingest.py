import abc
import csv
import glob
import itertools
import json
import os
import re
import sys
import typing
from datetime import datetime
from functools import cache
from multiprocessing.pool import ThreadPool
from pathlib import Path

import geopandas as gpd
import peewee as pw
import rasterio
import rich_click as click
import shapely as shp
import sshtunnel
from playhouse.postgres_ext import JSONField, PostgresqlExtDatabase
from playhouse.shortcuts import model_to_dict
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import cacafo.data.source
import cacafo.db.models as m
import cacafo.naip

# TODO fix this import
from cacafo.db.models import *


def check_if_populated(model):
    if model.select().count() != 0:
        raise ValueError(
            f"Table {model.__name__} is already populated;"
            "pass `overwrite=True` to wipe and repopulate, and `add=True`"
            "to add to existing data."
        )


def preflight(model, overwrite=False, add=False):
    if add:
        return
    if force:
        model.delete().execute()
    check_if_populated(model)


INGESTORS: dict[pw.Model, list[typing.Callable]] = {}


def ingestor(model):
    def decorator(func):
        if model not in INGESTORS:
            INGESTORS[model] = []
        INGESTORS[model].append(func)
        return func

    return decorator


@ingestor(County)
@ingestor(CountyGroup)
def counties(overwrite=False, add=False):
    preflight(County, overwrite=overwrite, add=add)
    preflight(CountyGroup, overwrite=overwrite, add=add)

    with open(cacafo.data.source.get("California_Counties.csv")) as f:
        counties = list(csv.DictReader(f))
    with open(cacafo.data.source.get("county_groups.csv")) as f:
        reader = csv.DictReader(f)
        county_groups_mappings = {row["County"]: row["Group Name"] for row in reader}
    CountyGroup.bulk_create(
        [CountyGroup(name=name) for name in county_groups_mappings.values]
    )
    county_groups = {
        name: CountyGroup.get(name=name) for name in county_groups_mappings.values()
    }
    County.bulk_create(
        [
            County(
                name=county["Name"],
                latitude=county["Latitude"],
                longitude=county["Longitude"],
                geometry=shp.from_wkt(county["the_geom"]),
                county_group=county_groups[county_groups_mappings[county["Name"]]],
            )
            for county in counties
        ]
    )


@ingestor(Parcel)
def parcels(overwrite=False, add=False):
    preflight(Parcel, overwrite=overwrite, add=add)

    parcel_dirs = (
        cacafo.data.source.get("parcels/polygon-centroids-merge-v2"),
        cacafo.data.source.get("afo-permits-geocoded"),
    )

    def add_from_directory(cls, directory):
        files = glob.glob(f"{directory}/*.csv")
        created = {
            (p.numb, p.county): p for p in pw.prefetch(cls.select(), County.select())
        }
        to_create = []
        for path in tqdm(files):
            with open(path) as f:
                rows = list(csv.DictReader(f))
            fname = os.path.basename(path)
            county = (
                County.select()
                .where(
                    County.name
                    == " ".join(fname.split(".")[0].split("-")[3:-2]).title()
                )
                .first()
            )
            for row in rows:
                if not row["parcelnumb"]:
                    continue
                parcel = cls(
                    county=county,
                    numb=row["parcelnumb"],
                    owner=row["owner"],
                    address=row["address"],
                    data=row,
                )
                if (parcel.numb, parcel.county) in created:
                    if created[(parcel.numb, parcel.county)].owner != parcel.owner:
                        print(
                            "Warning: multiple owners for parcel",
                            parcel.numb,
                            parcel.county,
                        )
                        print(created[(parcel.numb, parcel.county)].owner, parcel.owner)
                    if created[(parcel.numb, parcel.county)].address != parcel.address:
                        print(
                            "Warning: multiple addresses for parcel",
                            parcel.numb,
                            parcel.county,
                        )
                        print(
                            created[(parcel.numb, parcel.county)].address,
                            parcel.address,
                        )
                    continue
                created[(parcel.numb, parcel.county)] = parcel
                to_create.append(parcel)
                while to_create:
                    cls.bulk_create(to_create[:2000])
                    to_create = to_create[2000:]

    add_from_directory(parcel_dirs[0])
    add_from_directory(parcel_dirs[1])


@ingestor(RegulatoryProgram)
def regulatory_programs(overwrite=False, add=False):
    preflight(RegulatoryProgram, overwrite=overwrite, add=add)
    with open(cacafo.data.source.get("AFO_permits_all.csv")) as f:
        permits = list(csv.DictReader(f))
    unique_programs = set()
    for permit in permits:
        for program in permit["Program"].split(","):
            unique_programs.add(program.strip())
    for program in unique_programs:
        if program and str(program) != "nan":
            cls.create(name=program)


@ingestor(PermittedLocation)
def permitted_locations(overwrite=False, add=False):
    preflight(PermittedLocation, overwrite=overwrite, add=add)
    to_create = {}
    with open(cacafo.data.source.get("AFO_permits_all.csv")) as f:
        data = list(csv.DictReader(f))
    for row in tqdm(data):
        if (
            row["Latitude"]
            and row["Longitude"]
            and row["Latitude"] != "Error"
            and row["Longitude"] != "Error"
        ):
            to_create[(row["Latitude"], row["Longitude"])] = cls(
                latitude=row["Latitude"],
                longitude=row["Longitude"],
                geometry=shp.Point(float(row["Longitude"]), float(row["Latitude"])),
            )

    directory = cacafo.data.source.get("afo-permits-geocoded")
    files = glob.glob(f"{directory}/*.csv")
    data = []
    for path in tqdm(files):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        data.extend(rows)
    for row in tqdm(data):
        if (
            row["lat"]
            and row["lon"]
            and row["lat"] != "Error"
            and row["lon"] != "Error"
        ):
            to_create[(row["lat"], row["lon"])] = cls(
                latitude=row["lat"],
                longitude=row["lon"],
                geometry=shp.Point(float(row["lon"]), float(row["lat"])),
            )
    to_create = list(to_create.values())
    while to_create:
        cls.bulk_create(to_create[:2000])
        to_create = to_create[2000:]


def add_parcel_field_to_permits():
    county_name_to_id = {county.name: county.id for county in County.select()}
    parcelnumb_to_id = {
        (parcel.numb, parcel.county.id): parcel.id for parcel in Parcel.select()
    }
    permit_to_permit_data_location = {
        permit["id"]: (permit["latitude"], permit["longitude"])
        for permit in Permit.select(
            Permit.id, PermittedLocation.latitude, PermittedLocation.longitude
        )
        .join(PermitPermittedLocation)
        .join(PermittedLocation)
        .where(PermitPermittedLocation.source == "permit data")
        .dicts()
    }
    permit_to_geocoded_location = {
        permit["id"]: (permit["latitude"], permit["longitude"])
        for permit in Permit.select(
            Permit.id, PermittedLocation.latitude, PermittedLocation.longitude
        )
        .join(PermitPermittedLocation)
        .join(PermittedLocation)
        .where(PermitPermittedLocation.source == "address geocoding")
        .dicts()
    }
    permit_to_location = permit_to_geocoded_location | permit_to_permit_data_location
    location_to_permits = {}
    for permit_id, location in permit_to_location.items():
        location_to_permits.setdefault(location, []).append(permit_id)

    permit_parcel_rows = [
        row for row in csv.DictReader(open(data.get("permits_parcels.csv")))
    ]
    id_to_permit = {permit.id: permit for permit in Permit.select()}
    to_create = {}
    for row in permit_parcel_rows:
        if not row["parcelnumb"]:
            continue
        if not row["County"]:
            county = County.geocode(lon=row["Longitude"], lat=row["Latitude"])
            row["County"] = county.name
        if (
            row["parcelnumb"],
            county_name_to_id[row["County"]],
        ) not in parcelnumb_to_id:
            parcel = Parcel(
                numb=row["parcelnumb"],
                county=county_name_to_id[row["County"]],
                owner=row["owner"],
                address=row["address"],
                data={k: v for k, v in row.items() if k[0].islower()},
            )
            to_create[(row["parcelnumb"], county_name_to_id[row["County"]])] = parcel
    Parcel.bulk_create(to_create.values())
    parcelnumb_to_id = {
        (parcel.numb, parcel.county.id): parcel.id for parcel in Parcel.select()
    }
    to_update = []
    n_locations_not_found = 0
    for row in permit_parcel_rows:
        if not row["parcelnumb"]:
            continue
        location = (float(row["Latitude"]), float(row["Longitude"]))
        parcel_id = parcelnumb_to_id[
            (row["parcelnumb"], county_name_to_id[row["County"]])
        ]
        if location not in location_to_permits:
            print(f"Location not found: {location}")
            n_locations_not_found += 1
            continue
        for permit_id in location_to_permits[location]:
            id_to_permit[permit_id].parcel = parcel_id
            to_update.append(id_to_permit[permit_id])
    print(f"Locations not found: {n_locations_not_found}")
    Permit.bulk_update(to_update, fields=[Permit.parcel])
    return Permit.select().where(Permit.parcel.is_null()).count()


def add_facility_field_to_permits():
    from cacafo.cluster.permits import parcel_then_distance_matches

    facility_to_permits = parcel_then_distance_matches()
    permits = {permit.id: permit for permit in Permit.select()}
    to_update = []
    for facility, permit_ids in facility_to_permits.items():
        for permit_id in permit_ids:
            permits[permit_id].facility = facility
            to_update.append(permits[permit_id])
    Permit.bulk_update(to_update, fields=[Permit.facility])


@ingestor(Permit)
@ingestor(PermitPermittedLocation)
@ingestor(PermitRegulatoryProgram)
def permits(overwrite=False, add=False):
    preflight(Permit, overwrite=overwrite, add=add)
    preflight(PermitPermittedLocation, overwrite=overwrite, add=add)
    preflight(PermitRegulatoryProgram, overwrite=overwrite, add=add)

    with open(cacafo.data.source.get("AFO_permits_all.csv")) as f:
        data = list(csv.DictReader(f))

    directory = cacafo.data.source.get("afo-permits-geocoded")
    files = glob.glob(f"{directory}/*.csv")
    location_data = []
    for path in tqdm(files):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        location_data.extend(rows)
    index_to_location = {
        int(row["permit_id"]): (float(row["lat"]), float(row["lon"]))
        for row in location_data
    }

    permit_permitted_locations_to_create = []
    permit_regulatory_programs_to_create = []

    for i, row in enumerate(tqdm(data)):
        geocoded_lat_lon = index_to_location.get(i, None)
        try:
            geocoded_locations = list(
                PermittedLocation.select().where(
                    (PermittedLocation.latitude == geocoded_lat_lon[0])
                    & (PermittedLocation.longitude == geocoded_lat_lon[1])
                )
            )
        except (ValueError, TypeError, pw.DataError):
            geocoded_locations = []
        try:
            permitted_locations = list(
                PermittedLocation.select().where(
                    (PermittedLocation.latitude == row["Latitude"])
                    & (PermittedLocation.longitude == row["Longitude"])
                )
            )
        except (ValueError, pw.DataError):
            permitted_locations = []
        if not (permitted_locations or geocoded_locations):
            print("No locations for permit", row["WDID"])
        pop = None
        try:
            pop = int(float(row["Cafo Population"]))
        except ValueError:
            pass
        permit = cls.create(
            wdid=row["WDID"],
            agency_name=row["Agency"],
            agency_address=row["Agency Address"],
            facility_name=row["Facility Name"],
            facility_address=row["Facility Address"],
            permitted_population=pop,
            regulatory_measure_status=row["Regulatory Measure Status"],
            data=row,
        )
        for location in permitted_locations:
            permit_permitted_locations_to_create.append(
                PermitPermittedLocation(
                    permit=permit,
                    permitted_location=location,
                    source="permit data",
                )
            )
        for location in geocoded_locations:
            permit_permitted_locations_to_create.append(
                PermitPermittedLocation(
                    permit=permit,
                    permitted_location=location,
                    source="address geocoding",
                )
            )

        regulatory_program_strings = [p.strip() for p in row["Program"].split(",")]
        regulatory_programs = list(
            RegulatoryProgram.select().where(
                RegulatoryProgram.name.in_(regulatory_program_strings)
            )
        )

        for program in regulatory_programs:
            permit_regulatory_programs_to_create.append(
                PermitRegulatoryProgram(
                    permit=permit,
                    regulatory_program=program,
                )
            )
    while permit_permitted_locations_to_create:
        PermitPermittedLocation.bulk_create(permit_permitted_locations_to_create[:2000])
        permit_permitted_locations_to_create = permit_permitted_locations_to_create[
            2000:
        ]
    while permit_regulatory_programs_to_create:
        PermitRegulatoryProgram.bulk_create(permit_regulatory_programs_to_create[:2000])
        permit_regulatory_programs_to_create = permit_regulatory_programs_to_create[
            2000:
        ]

    add_parcel_field_to_permits()


@ingestor(FacilityAnimalType)
def facility_animal_types(data_path=None, overwrite=False, add=False):
    preflight(FacilityAnimalType, overwrite=overwrite, add=add)
    csv_paths = []
    if data_path is None:
        csv_paths = [
            cacafo.data.source.get(file)
            for file in cacafo.data.source.list()
            if "new_animal_typing/" in file
        ]
    else:
        if os.path.exists(data_path):
            csv_paths = [data_path]
        else:
            csv_paths = [cacafo.data.source.get(data_path)]
    animal_types = {at.name: at.id for at in m.AnimalType.select()}
    animal_types["cattle"] = animal_types["cows"]
    animal_types["swine"] = animal_types["pigs"]
    rows = []
    for file in csv_paths:
        with open(file, "r") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))

    from cacafo.geom import clean_facility_geometry

    facilities = {
        facility.id: facility
        for facility in pw.prefetch(
            m.Facility.select()
            .join(m.Building, on=(m.Facility.id == m.Building.facility_id))
            .distinct(),
            m.Building.select(),
        )
    }
    facility_uuid_to_id = {
        str(facility.uuid): facility.id for facility in facilities.values()
    }
    # do bounding box
    facilities = {
        int(k): clean_facility_geometry(v).convex_hull for k, v in facilities.items()
    }
    ids = sorted(facilities.keys())
    geoms = [f[1] for f in sorted(facilities.items(), key=lambda x: x[0])]
    tree = shp.STRtree(geoms)
    to_create = []
    for row in rows:
        if row["uuid"] in facility_uuid_to_id:
            row["facility_id"] = facility_uuid_to_id[row["uuid"]]
        else:
            lon = row.get("longitude", row.get("lon"))
            lat = row.get("latitude", row.get("lat"))
            latlon = shp.geometry.Point((float(lon), float(lat)))
            outputs = tree.query(latlon, predicate="intersects")
            if len(outputs) == 0:
                print(f"Facility not found: {row}")
                continue
            if len(outputs) > 1:
                print(f"Facility ambiguous: {row}")
                continue
            row["facility_id"] = ids[outputs[0]]
        if row["isAFO?"].strip() == "n":
            to_create.append(
                m.FacilityAnimalType(
                    facility=row["facility_id"],
                    animal_type=animal_types["not afo"],
                    label_source="human",
                )
            )
        if row["isCAFO?"].strip() == "n":
            to_create.append(
                m.FacilityAnimalType(
                    facility=row["facility_id"],
                    animal_type=animal_types["not cafo"],
                    label_source="human",
                )
            )
        if row["notes"] == "feedlot":
            to_create.append(
                m.FacilityAnimalType(
                    facility=row["facility_id"],
                    animal_type=animal_types["feedlot"],
                    label_source="human",
                )
            )
        if row["subtype"]:
            to_create.append(
                m.FacilityAnimalType(
                    facility=row["facility_id"],
                    animal_type=animal_types[row["subtype"].strip()],
                    label_source="human",
                )
            )
        if row["animal type"]:
            if row["animal type"] not in animal_types:
                animal_type = m.AnimalType(name=row["animal type"].strip())
                animal_type.save()
                animal_types[row["animal type"]] = animal_type.id
            to_create.append(
                m.FacilityAnimalType(
                    facility=row["facility_id"],
                    animal_type=animal_types[row["animal type"]],
                    label_source="human",
                )
            )
    # remove duplicates within to_create
    to_create = list(
        {(fat.facility_id, fat.animal_type_id): fat for fat in to_create}.values()
    )
    # remove all existing animal types for these facilities, because they are superceded
    m.FacilityAnimalType.delete().where(
        m.FacilityAnimalType.facility_id.in_([fat.facility_id for fat in to_create])
    ).execute()
    m.FacilityAnimalType.bulk_create(to_create)
    return to_create
