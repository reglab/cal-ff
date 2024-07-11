import csv
import datetime
import glob
import itertools
import json
import os
import re
import sys
from multiprocessing.pool import ThreadPool

import geopandas as gpd
import models as m
import peewee as pw
import playhouse.sqlite_ext as pwext
import shapely as shp
from playhouse.shortcuts import model_to_dict
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import data
import naip


class ParcelLocation(pw.Model):
    parcel_id = pw.IntegerField()
    geometry = m.WktField()

    class Meta:
        database = m.db


def populate_building_parcels(path_to_parcel_csvs):
    csvs = glob.glob(f"{path_to_parcel_csvs}/*.csv")
    parcel_locations = []
    ParcelLocation.create_table(temporary=True)

    parcel_numb_and_county_to_id = {
        (parcel.numb, parcel.county.name): parcel.id
        for parcel in pw.prefetch(m.Parcel.select(), m.County.select())
    }

    for csvfile in tqdm(csvs):
        print(os.path.basename(csvfile))
        county_name = " ".join(
            os.path.basename(csvfile).split(".")[0].split("-")[3:-2]
        ).title()
        with open(csvfile, "r") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            location = shp.Point(float(row["longitude"]), float(row["latitude"]))
            if not (row["parcelnumb"], county_name) in parcel_numb_and_county_to_id:
                print(f"Parcel not found: {row['parcelnumb']}, {county_name}")
                continue
            parcel_id = parcel_numb_and_county_to_id[(row["parcelnumb"], county_name)]
            parcel_locations.append({"parcel_id": parcel_id, "geometry": location})
    with m.db.atomic():
        ParcelLocation.insert_many(parcel_locations).execute()
    # add spatial index
    try:
        ParcelLocation.raw(
            "CREATE INDEX parcel_location_geom_idx ON parcellocation USING GIST(geometry)"
        ).execute()
    except pw.ProgrammingError:
        breakpoint()
    matches = (
        m.Building.select(
            m.Building.id.alias("building_id"),
            ParcelLocation.parcel_id,
        )
        .join(
            ParcelLocation,
            on=pw.fn.ST_Contains(m.Building.geometry, ParcelLocation.geometry),
        )
        .where(
            m.Building.parcel.is_null(),
        )
        .tuples()
    )
    to_save = []
    return matches
    for match in matches:
        import ipdb

        ipdb.set_trace()
        match.parcel = match.parcel_id
        del match.parcel_id
        to_save.append(match)
    m.Building.bulk_update(to_save, fields=[m.Building.parcel])


def load_image_bucket(image_bucket_csv_path):
    images = {image.name: image for image in m.Image.select()}
    with open(image_bucket_csv_path, "r") as f:
        image_buckets = {row["image_name"]: row["bucket"] for row in csv.DictReader(f)}
    to_save = []
    for image in tqdm(images):
        if image in image_buckets:
            images[image].bucket = image_buckets[image]
            to_save.append(images[image])
        if len(to_save) > 10000:
            m.Image.bulk_update(to_save, fields=[m.Image.bucket])
            to_save = []


def load_adjacent_status(path_to_json_annotations):
    json_files = glob.glob(f"{path_to_json_annotations}/*.json")
    names = set()
    for file in tqdm(json_files):
        name = os.path.basename(file).split(".")[0]
        names.add(name)
    m.Image.update(label_status="adjacent").where(m.Image.name.in_(names)).execute()


def migrate_buildings_to_image_fkey():
    try:
        m.Building.raw(
            "ALTER TABLE building ADD COLUMN image_id INTEGER REFERENCES image(id)"
        ).execute()
    except pw.ProgrammingError as e:
        if 'column "image_id" of relation "building" already exists' in str(e):
            pass
        else:
            raise
    buildings = m.Building.select(
        m.Building.id,
        m.Image.id,
    ).join(m.Image)
    for building in tqdm(buildings):
        building.image = int(building.image.id)
    m.Building.bulk_update(buildings, fields=[m.Building.image])


def set_removed_image_label_status():
    bulk = []
    for image_name in tqdm(naip.list_removed_images()):
        bulk.append(image_name)
        if len(bulk) > 1000:
            m.Image.update(label_status="removed").where(
                m.Image.name.in_(bulk)
            ).execute()
            bulk = []
    return (
        m.Image.update(label_status="removed")
        .where(m.Image.name.in_(bulk), m.Image.label_status.is_null())
        .execute()
    )


def set_active_learner_image_label_status():
    bulk = []
    with open(data.get("labels/annotated.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bulk.append(row["jpeg_path"].split(".")[0])
            if len(bulk) > 1000:
                m.Image.update(label_status="active learner").where(
                    m.Image.name.in_(bulk)
                ).execute()
                bulk = []
        m.Image.update(label_status="active learner").where(
            m.Image.name.in_(bulk)  # & m.Image.label_status.is_null()
        ).execute()
    return m.Image.select().where(m.Image.label_status == "active learner").count()


def set_permit_image_label_status():
    active_permits = glob.glob(data.get("labels/missed_active_permits") + "/*.json")
    historical_permits = glob.glob(
        data.get("labels/missed_historical_permits") + "/*.json"
    )
    bulk = []
    for permit in itertools.chain(active_permits, historical_permits):
        bulk.append(os.path.basename(permit).split(".")[0])
        if len(bulk) > 1000:
            m.Image.update(label_status="permit").where(
                m.Image.name.in_(bulk)
            ).execute()
            bulk = []
    m.Image.update(label_status="permit").where(
        m.Image.name.in_(bulk) & m.Image.label_status.is_null()
    ).execute()
    return m.Image.select().where(m.Image.label_status == "permit").count()


def set_adjacent_image_label_status():
    adjacent = glob.glob(data.get("labels/adjacent") + "/*.json")
    bulk = []
    for annotation in adjacent:
        bulk.append(os.path.basename(annotation).split(".")[0])
        if len(bulk) > 1000:
            m.Image.update(label_status="adjacent").where(
                m.Image.name.in_(bulk) & m.Image.label_status.is_null()
            ).execute()
            bulk = []
    m.Image.update(label_status="adjacent").where(
        m.Image.name.in_(bulk) & m.Image.label_status.is_null()
    ).execute()
    return m.Image.select().where(m.Image.label_status == "adjacent").count()


def set_unlabeled_image_label_status():
    m.Image.update(label_status="unlabeled").where(
        m.Image.label_status.is_null()
    ).execute()
    return m.Image.select().where(m.Image.label_status == "unlabeled").count()


def set_label_statuses():
    m.Image.update(label_status=None).execute()
    set_removed_image_label_status()
    set_active_learner_image_label_status()
    set_permit_image_label_status()
    set_adjacent_image_label_status()
    set_unlabeled_image_label_status()


def add_geometry_to_permittedlocation():
    m.PermittedLocation.raw(
        "ALTER TABLE permittedlocation ADD COLUMN geometry GEOMETRY"
    ).execute()
    permitted_locations = m.PermittedLocation.select()
    for permitted_location in permitted_locations:
        permitted_location.geometry = shp.Point(
            permitted_location.longitude, permitted_location.latitude
        )
    m.PermittedLocation.bulk_update(
        permitted_locations, fields=[m.PermittedLocation.geometry]
    )
    try:
        m.PermittedLocation.raw(
            "CREATE INDEX permitted_location_geom_idx ON permittedlocation USING GIST(geometry)"
        ).execute()
    except pw.ProgrammingError:
        breakpoint()


def load_raw_cloud_factory_image_annotations_from_list(annotations):
    # trim, e.g. "/Riverside.jpeg"
    image_names = [
        annotations["filename"].strip("/").split(".")[0] for annotations in annotations
    ]
    images = {
        image.name: image
        for image in m.Image.select().where(m.Image.name.in_(image_names))
    }
    db_annotations = []
    for annotation in annotations:
        db_annotations.append(
            m.RawCloudFactoryImageAnnotation(
                image=images[annotation["filename"].strip("/").split(".")[0]],
                created_at=annotation["createdAt"],
                json=annotation,
            )
        )
    while db_annotations:
        m.RawCloudFactoryImageAnnotation.bulk_create(db_annotations[:1000])
        db_annotations = db_annotations[1000:]


def clear_adjacent_image_flags():
    descriptions = data.get("adjacent_image_flag_descriptions.csv")

    def remove_flag(json):
        json["annotations"] = [
            annotation
            for annotation in json["annotations"]
            if annotation["label"] != "flag"
        ]
        return json

    def remove_cafo(json):
        json["annotations"] = [
            annotation
            for annotation in json["annotations"]
            if annotation["label"] != "cafo"
        ]
        return json

    def add_blank(json):
        json["annotations"].append(
            {
                "label": "Blank",
                "creator": "vim@law.stanford.edu",
                "createdAt": datetime.datetime.now().isoformat(),
                "metadata": {
                    "note": "This image was manually flagged as blank, after CF labeling.",
                },
            }
        )
        return json

    with open(descriptions, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        annotation_names = [row["file"].split(".")[0] for row in rows]
        annotations = {
            annotation.image.name: annotation
            for annotation in m.RawCloudFactoryImageAnnotation.select(
                m.RawCloudFactoryImageAnnotation.json,
                m.Image.name,
                m.RawCloudFactoryImageAnnotation.id,
            )
            .join(m.Image)
            .where(m.Image.name.in_(annotation_names))
        }
        for row in tqdm(rows):
            action_code = int(row[""])
            json = annotations[row["file"].split(".")[0]].json
            match action_code:
                case 0:
                    pass
                case 1 | 2:
                    annotations[row["file"].split(".")[0]].json = remove_flag(json)
                case 3 | 4 | 6:
                    annotations[row["file"].split(".")[0]].json = add_blank(
                        remove_cafo(remove_flag(json))
                    )
                case 5:
                    pass
                case _:
                    raise ValueError(f"Invalid action code: {action_code}")
        m.RawCloudFactoryImageAnnotation.bulk_update(
            annotations.values(), fields=[m.RawCloudFactoryImageAnnotation.json]
        )


def load_adjacents_as_raw_cloud_factory_image_annotations():
    json_files = glob.glob(data.get("labels/adjacent") + "/*.json")
    annotations = []
    for json_file in tqdm(json_files):
        with open(json_file, "r") as f:
            j = json.load(f)
            annotations.append(j)
    load_raw_cloud_factory_image_annotations_from_list(annotations)


def load_jsonl_as_raw_cloud_factory_image_annotations(jsonl_path):
    annotations = []
    with open(jsonl_path, "r") as f:
        for line in f:
            j = json.loads(line)
            annotations.append(j)
    load_raw_cloud_factory_image_annotations_from_list(annotations)


def load_ex_ante_permits_as_raw_cloud_factory_image_annotations():
    ex_ante_permit_jsonl = data.get("labels/ex_ante_permits.jsonl")
    load_jsonl_as_raw_cloud_factory_image_annotations(ex_ante_permit_jsonl)


def load_ex_ante_permit_adjacents_as_raw_cloud_factory_image_annotations():
    ex_ante_permit_adjacents = data.get("labels/ex_ante_permit_adjacents.jsonl")
    load_jsonl_as_raw_cloud_factory_image_annotations(ex_ante_permit_adjacents)


def load_buildings_from_raw_cloud_factory_image_annotations():
    annotations = (
        m.RawCloudFactoryImageAnnotation.select(
            m.RawCloudFactoryImageAnnotation.json.alias("json"),
        )
        .join(m.Image)
        .join(m.Building, pw.JOIN.LEFT_OUTER, on=m.Building.image)
        .where(
            (m.Building.id.is_null())
            & (
                m.RawCloudFactoryImageAnnotation.json["annotations"][0]["label"]
                != "Blank"
            )
        )
        .tuples()
    )
    annotations = sum(annotations, ())
    # threads are helpful because we sometimes have to do io bound image download
    threads = 4
    buildings = []
    pbar = tqdm(total=len(annotations))

    def construct_annotations(annotations):
        for annotation in annotations:
            buildings.extend(m.Building.construct_from_annotation_dict(annotation))
            pbar.update(1)

    with ThreadPool(threads) as pool:
        pool.map(
            construct_annotations,
            [annotations[i::threads] for i in range(threads)],
        )
    building_conflicts = set(
        sum(
            list(
                m.Building.select(m.Building.geometry)
                .where(
                    m.Building.geometry.in_(
                        [building.geometry for building in buildings]
                    )
                )
                .tuples()
            ),
            (),
        )
    )

    buildings = [
        building
        for building in buildings
        if building.geometry not in building_conflicts
    ]
    m.Building.bulk_create(buildings)


def load_permit_ex_ante_status():
    ex_ante_permit_jsonl = data.get("labels/ex_ante_permits.jsonl")
    images_to_update = []
    with open(ex_ante_permit_jsonl, "r") as f:
        for line in f:
            j = json.loads(line)
            image = m.Image.get(m.Image.name == j["filename"].split(".")[0].strip("/"))
            image.label_status = "ex_ante_permit"
            images_to_update.append(image)
    m.Image.bulk_update(images_to_update, fields=[m.Image.label_status])


def load_ex_ante_permit_adjacent_status():
    ex_ante_permit_adjacents_jsonl = data.get("labels/ex_ante_permit_adjacents.jsonl")
    images_to_update = []
    with open(ex_ante_permit_adjacents_jsonl, "r") as f:
        for line in f:
            j = json.loads(line)
            image = m.Image.get(m.Image.name == j["filename"].split(".")[0].strip("/"))
            image.label_status = "adjacent"
            images_to_update.append(image)
    m.Image.bulk_update(images_to_update, fields=[m.Image.label_status])


def load_missed_adjacent_status():
    ex_ante_permit_adjacents_jsonl = data.get("labels/missed_adjacents.jsonl")
    images_to_update = []
    with open(ex_ante_permit_adjacents_jsonl, "r") as f:
        for line in f:
            j = json.loads(line)
            image = m.Image.get(m.Image.name == j["filename"].split(".")[0].strip("/"))
            image.label_status = "adjacent"
            images_to_update.append(image)
    m.Image.bulk_update(images_to_update, fields=[m.Image.label_status])


def load_non_med_recall_status():
    ex_ante_permit_adjacents_jsonl = data.get("labels/non_med_recall.jsonl")
    images_to_update = []
    with open(ex_ante_permit_adjacents_jsonl, "r") as f:
        for line in f:
            j = json.loads(line)
            image = m.Image.get(m.Image.name == j["filename"].split(".")[0].strip("/"))
            image.label_status = "active learner"
            images_to_update.append(image)
    m.Image.bulk_update(images_to_update, fields=[m.Image.label_status])


def recluster():
    m.BuildingRelationship.load()
    m.Facility.load()


def add_image_stratum():
    try:
        m.Image.raw("ALTER TABLE image ADD COLUMN stratum TEXT").execute()
    except pw.ProgrammingError:
        pass

    completed_buckets = ["1.25", "1.75", "3", "inf", "ex ante permit"]

    m.Image.update(stratum=None).execute()
    for county_group in tqdm(m.CountyGroup.select(), total=13):
        for bucket in ("0", "1"):
            m.Image.update(stratum=f"{bucket}:{county_group.name}").where(
                m.Image.id.in_(
                    m.Image.select(m.Image.id)
                    .join(m.County)
                    .join(m.CountyGroup)
                    .where(m.CountyGroup.id == county_group.id)
                )
                & (m.Image.bucket == bucket)
            ).execute()
    m.Image.update(stratum="completed").where(
        m.Image.bucket.in_(completed_buckets)
    ).execute()
    img = m.Image.alias()
    positive_images = m.Image.select().join(m.Building).distinct()
    img = m.Image.alias()
    (
        m.Image.update(stratum="completed")
        .where(
            m.Image.id.in_(
                m.Image.select(m.Image.id)
                .join(m.ImageAdjacency, on=(m.Image.id == m.ImageAdjacency.image_id))
                .join(img, on=(m.ImageAdjacency.adjacent_image_id == img.id))
                .where(
                    img.bucket.in_(completed_buckets)
                    & (img.id.in_(positive_images))
                    & (img.label_status != "removed")
                    & (m.Image.label_status != "removed")
                )
            )
        )
        .execute()
    )
    m.Image.update(stratum="post hoc").where(
        (m.Image.stratum != "completed")
        & (m.Image.label_status.in_(["adjacent", "post hoc permit"]))
    ).execute()
    print("set other adjacent")


def add_parcel_field_to_permits():
    try:
        m.Permit.raw(
            "ALTER TABLE permit ADD COLUMN parcel_id INTEGER REFERENCES parcel(id)"
        ).execute()
    except pw.ProgrammingError:
        pass

    county_name_to_id = {county.name: county.id for county in m.County.select()}
    parcelnumb_to_id = {
        (parcel.numb, parcel.county.id): parcel.id for parcel in m.Parcel.select()
    }
    permit_to_permit_data_location = {
        permit["id"]: (permit["latitude"], permit["longitude"])
        for permit in m.Permit.select(
            m.Permit.id, m.PermittedLocation.latitude, m.PermittedLocation.longitude
        )
        .join(m.PermitPermittedLocation)
        .join(m.PermittedLocation)
        .where(m.PermitPermittedLocation.source == "permit data")
        .dicts()
    }
    permit_to_geocoded_location = {
        permit["id"]: (permit["latitude"], permit["longitude"])
        for permit in m.Permit.select(
            m.Permit.id, m.PermittedLocation.latitude, m.PermittedLocation.longitude
        )
        .join(m.PermitPermittedLocation)
        .join(m.PermittedLocation)
        .where(m.PermitPermittedLocation.source == "address geocoding")
        .dicts()
    }
    permit_to_location = permit_to_geocoded_location | permit_to_permit_data_location
    location_to_permits = {}
    for permit_id, location in permit_to_location.items():
        location_to_permits.setdefault(location, []).append(permit_id)

    permit_parcel_rows = [
        row for row in csv.DictReader(open(data.get("permits_parcels.csv")))
    ]
    id_to_permit = {permit.id: permit for permit in m.Permit.select()}
    to_create = {}
    for row in permit_parcel_rows:
        if not row["parcelnumb"]:
            continue
        if not row["County"]:
            county = m.County.geocode(lon=row["Longitude"], lat=row["Latitude"])
            row["County"] = county.name
        if (
            row["parcelnumb"],
            county_name_to_id[row["County"]],
        ) not in parcelnumb_to_id:
            parcel = m.Parcel(
                numb=row["parcelnumb"],
                county=county_name_to_id[row["County"]],
                owner=row["owner"],
                address=row["address"],
                data={k: v for k, v in row.items() if k[0].islower()},
            )
            to_create[(row["parcelnumb"], county_name_to_id[row["County"]])] = parcel
    m.Parcel.bulk_create(to_create.values())
    parcelnumb_to_id = {
        (parcel.numb, parcel.county.id): parcel.id for parcel in m.Parcel.select()
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
    m.Permit.bulk_update(to_update, fields=[m.Permit.parcel])
    return m.Permit.select().where(m.Permit.parcel.is_null()).count()


def add_facility_field_to_permits(min_facility_dist_diff=20):
    try:
        m.Permit.raw(
            "ALTER TABLE permit ADD COLUMN facility_id INTEGER REFERENCES facility(id)"
        ).execute()
    except pw.ProgrammingError:
        pass

    permit_facility_parcel_matches = {
        permit["id"]: permit["facility"]
        for permit in m.Permit.select(m.Permit.id, m.Building.facility_id)
        .join(m.Building, on=(m.Permit.parcel_id == m.Building.parcel_id))
        .where(m.Building.cafo)
        .dicts()
    }

    to_update = []
    id_to_permit = {permit.id: permit for permit in m.Permit.select()}
    for permit_id in id_to_permit:
        if permit_id not in permit_facility_parcel_matches:
            continue
        facility_id = permit_facility_parcel_matches[permit_id]
        id_to_permit[permit_id].facility = facility_id
        to_update.append(id_to_permit[permit_id])

    m.Permit.bulk_update(to_update, fields=[m.Permit.facility])

    # ties are 20m or less.
    # 1. match permit to facility w all parcel matches
    # 2. select permits with null facilities
    # 3. closest facility is matched to remaining under 1km unless another facility is within 20m of the first closest
    # 4. all other nulls are hand coded. (only need to hand code ones that aren't hand coded)

    to_update = []
    id_to_permit = {
        permit.id: permit
        for permit in m.Permit.select().where(m.Permit.facility.is_null())
    }
    unresolvable_facility_matches = 0

    for permit_id in id_to_permit:
        permit_facility_distance = (
            m.Permit.select(
                m.Permit.id,
                m.FacilityPermittedLocation.facility_id,
                m.FacilityPermittedLocation.distance,
            )
            .join(
                m.PermitPermittedLocation,
                on=(m.Permit.id == m.PermitPermittedLocation.permit_id),
            )
            .join(
                m.FacilityPermittedLocation,
                on=(
                    m.PermitPermittedLocation.permitted_location_id
                    == m.FacilityPermittedLocation.permitted_location_id
                ),
            )
            .where(
                (m.Permit.id == permit_id)
                & (m.PermitPermittedLocation.source == "permit data")
            )
            .order_by(m.FacilityPermittedLocation.distance)
            .dicts()
        )
        if len(permit_facility_distance) == 0:
            continue

        min_dist = float("inf")
        facility_id = None
        for row in permit_facility_distance[:2]:
            if abs(row["distance"] - min_dist) < min_facility_dist_diff:
                facility_id = None
                unresolvable_facility_matches += 1
                break
            else:
                min_dist = row["distance"]
                facility_id = row["facility"]

        id_to_permit[permit_id].facility = facility_id
        to_update.append(id_to_permit[permit_id])
    print(
        "Number of permits with matching permits too close together",
        unresolvable_facility_matches,
    )
    m.Permit.bulk_update(to_update, fields=[m.Permit.facility])
    return m.Permit.select().where(m.Permit.facility.is_null()).count()


def add_missing_parcels_to_buildings():
    path = "source_data/missed_adjacents_parcels/"
    csvs = glob.glob(f"{path}/*.csv")
    parcelnumb_to_id = {
        (parcel.numb, parcel.county.id): parcel.id for parcel in m.Parcel.select()
    }
    to_update = []
    buildings = {
        building.id: building
        for building in m.Building.select().where(m.Building.parcel.is_null())
    }
    for csvfile in csvs:
        fname = os.path.basename(csvfile)
        county = (
            m.County.select()
            .where(
                m.County.name == " ".join(fname.split(".")[0].split("-")[3:-2]).title()
            )
            .first()
        )

        with open(csvfile, "r") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if not row["parcelnumb"]:
                continue
            if (row["parcelnumb"], county.id) not in parcelnumb_to_id:
                raise ValueError(f"Parcel not found: {row['parcelnumb']}")
            parcel_id = parcelnumb_to_id[(row["parcelnumb"], county.id)]
            buildings[int(row["id"])].parcel = parcel_id
            to_update.append(buildings[int(row["id"])])
    return m.Building.bulk_update(to_update, fields=[m.Building.parcel])


def add_new_animal_types():
    files = [data.get(file) for file in data.list() if "new_animal_typing/" in file]
    animal_types = {at.name: at.id for at in m.AnimalType.select()}
    animal_types["cattle"] = animal_types["cows"]
    animal_types["swine"] = animal_types["pigs"]
    rows = []
    for file in files:
        with open(file, "r") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    for row in rows:
        if "id" in row:
            row["facility_id"] = row["id"]
    facilities = {
        facility.id: facility
        for facility in pw.prefetch(
            m.Facility.select()
            # .join(m.FacilityAnimalType, pw.JOIN.LEFT_OUTER)
            .join(m.Building, on=(m.Facility.id == m.Building.facility_id))
            # .where(m.FacilityAnimalType.id.is_null())
            .distinct(),
            m.Building.select(),
        )
    }
    from reports import _clean_facility_geometry

    # do bounding box
    facilities = {
        int(k): _clean_facility_geometry(v).convex_hull for k, v in facilities.items()
    }
    ids = sorted(facilities.keys())
    geoms = [f[1] for f in sorted(facilities.items(), key=lambda x: x[0])]
    tree = shp.STRtree(geoms)
    to_create = []
    for row in rows:
        latlon = shp.geometry.Point((row["longitude"], row["latitude"]))
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
                    animal_type=animal_types[row["subtype"]],
                    label_source="human",
                )
            )
        if row["animal type"]:
            if row["animal type"] not in animal_types:
                animal_type = m.AnimalType(name=row["animal type"])
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


def delete_duplicate_facility_animal_types():
    facility_animal_types = {}
    for fat in m.FacilityAnimalType.select():
        key = (fat.facility_id, fat.animal_type_id)
        if key in facility_animal_types:
            facility_animal_types[key].append(fat)
        else:
            facility_animal_types[key] = [fat]
    to_delete = []
    for key, fats in facility_animal_types.items():
        if len(fats) == 1:
            continue
        if any(fat.label_source == "human" for fat in fats):
            if fats[0].label_source != "human":
                fats[0].label_source = "human"
                fats[0].save()
        to_delete.extend(fats[1:])
    m.FacilityAnimalType.delete().where(
        m.FacilityAnimalType.id.in_([fat.id for fat in to_delete])
    ).execute()


def add_permit_animal_types():
    from reports import parcel_then_distance_matches

    matches = parcel_then_distance_matches(200, cow_only=True)
    to_create = []
    already_typed = set(
        t[0]
        for t in m.FacilityAnimalType.select(
            m.FacilityAnimalType.facility_id.distinct(),
        )
        .where(m.FacilityAnimalType.animal_type == 1)
        .tuples()
    )
    for facility_id in matches:
        if facility_id in already_typed or facility_id is None:
            continue
        to_create.append(
            m.FacilityAnimalType(
                facility_id=facility_id,
                animal_type=1,
                label_source="permit",
            )
        )
    m.FacilityAnimalType.bulk_create(to_create)


if __name__ == "__main__":
    pass
