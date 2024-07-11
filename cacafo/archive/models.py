import csv
import glob
import itertools
import json
import os
import re
import sys
from datetime import datetime
from functools import cache
from multiprocessing.pool import ThreadPool

import geopandas as gpd
import peewee as pw
import rasterio
import shapely as shp
import sshtunnel
from playhouse.postgres_ext import JSONField, PostgresqlExtDatabase
from playhouse.shortcuts import model_to_dict

# from functools import cache
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import data
import naip

db = PostgresqlExtDatabase(
    "cacafo", user="vim", password="cacafo", host="0.0.0.0", port=55550
)
# db = pw.SqliteDatabase("ca_cafo_inventory.db", pragmas={"foreign_keys": "on"})

BUILDING_THRESHOLD_RELATIONSHIP_QUERY = open(
    "building_threshold_relationship_query.sql"
).read()


class WktField(pw.Field):
    field_type = "text"

    def db_value(self, value):
        if value is None:
            return value
        if isinstance(value, shp.Geometry):
            return value.wkt
        raise TypeError(f"A WktField requires shapely.lib.Geometry")

    def python_value(self, value):
        if value is None:
            return None
        return shp.from_wkt(value)


class CountyGroup(pw.Model):
    name = pw.TextField(unique=True)

    class Meta:
        database = db

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()
        with open(data.get("county_groups.csv")) as f:
            reader = csv.DictReader(f)
            names = {row["Group Name"] for row in reader}
        cls.bulk_create([cls(name=name) for name in names])


class County(pw.Model):
    name = pw.TextField(unique=True)
    latitude = pw.DoubleField()
    longitude = pw.DoubleField()
    geometry = WktField()
    county_group = pw.ForeignKeyField(CountyGroup, backref="counties", null=True)

    class Meta:
        database = db

    @classmethod
    def add_county_groups(cls):
        try:
            cls.raw("alter table county add column county_group_id integer").execute()
        except pw.ProgrammingError as e:
            if "already exists" not in str(e):
                raise e
        with open(data.get("county_groups.csv")) as f:
            reader = csv.DictReader(f)
            county_groups = {row["County"]: row["Group Name"] for row in reader}
        for county in cls.select():
            county.county_group = CountyGroup.get(name=county_groups[county.name])
            county.save()

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()

        csv.field_size_limit(sys.maxsize)

        with open("source_data/California_Counties.csv") as f:
            counties = list(csv.DictReader(f))
        for county in counties:
            cls.create(
                name=county["Name"],
                latitude=county["Latitude"],
                longitude=county["Longitude"],
                geometry=shp.from_wkt(county["the_geom"]),
            )
        # cannot create index because of size limitaitons

    @classmethod
    def load_from_dump(cls):
        cls.drop_table()
        cls.create_table()
        csv.field_size_limit(sys.maxsize)
        with open("outputs/counties.csv") as f:
            reader = csv.DictReader(f)
            to_create = []
            for row in tqdm(reader):
                to_create.append(
                    cls(**row | {"geometry": shp.from_wkt(row["geometry"])})
                )
            pbar = tqdm(total=len(to_create))
            while to_create:
                cls.bulk_create(to_create[:2000])
                to_create = to_create[2000:]
                pbar.update(2000)
            pbar.close()

    @classmethod
    def _counties(cls):
        if not hasattr(cls, "_cache_counties"):
            cls._cache_counties = list(cls.select())
        return cls._cache_counties

    @classmethod
    def geocode(cls, lon=None, lat=None):
        if None in (lon, lat):
            raise ValueError("None values given to goecode")
        point = shp.Point(lon, lat)
        for county in cls._counties():
            if county.geometry.contains(point):
                return county
        raise ValueError(f"No county found for {point}")


class Facility(pw.Model):
    latitude = pw.DoubleField()
    longitude = pw.DoubleField()
    geometry = WktField()
    uuid = pw.UUIDField(unique=True)

    class Meta:
        database = db

    @classmethod
    def _building_set_to_uuid(cls):
        building_set_to_uuid = {}
        building_to_uuid = {
            b["id"]: b["uuid"]
            for b in Building.select(Building.id, cls.uuid).join(cls).dicts()
        }
        uuid_to_building_set = {}
        for b, uuid in building_to_uuid.items():
            uuid_to_building_set.setdefault(uuid, set()).add(b)
        building_set_to_uuid = {
            frozenset(building_set): uuid
            for uuid, building_set in uuid_to_building_set.items()
        }
        return building_set_to_uuid

    @staticmethod
    def load(**kwargs):
        old_building_sets_to_uuids = {}
        if Facility.table_exists():
            old_building_sets_to_uuids = Facility._building_set_to_uuid()
        Building.update(facility=None).execute()
        Permit.update(facility=None).execute()
        # drop Faciulitypermitedlocation and facilityanimaltype
        FacilityPermittedLocation.drop_table()
        FacilityAnimalType.drop_table()
        ConstructionAnnotation.drop_table()
        Facility.delete().execute()

        try:
            Facility.raw(
                "alter table facility alter column uuid drop not null"
            ).execute()
        except pw.ProgrammingError as e:
            if "does not exist" not in str(e):
                raise e
        try:
            Facility.raw(
                "alter table building add constraint building_facility_fk_id foreign key (facility_id) references facility(id)"
            ).execute()
        except pw.ProgrammingError as e:
            if "already exists" not in str(e):
                raise e
        building_groups = list(BuildingRelationship.facilities(**kwargs))
        geoms = {
            row["id"]: row["geometry"]
            for row in Building.select(Building.id, Building.geometry).dicts()
        }
        to_create = []
        for building_ids in tqdm(building_groups):
            geom = shp.MultiPolygon(
                [
                    geoms[b].convex_hull
                    for b in building_ids
                    if isinstance(geoms[b].convex_hull, shp.Polygon)
                ]
            )
            if not geom:
                continue
            lat = geom.centroid.y
            lon = geom.centroid.x
            f = Facility(
                latitude=lat,
                longitude=lon,
                geometry=geom,
            )
            to_create.append(f)
        Facility.bulk_create(to_create)
        building_id_to_object = {b.id: b for b in Building.select()}
        for building_ids, facility in zip(building_groups, to_create):
            for b in building_ids:
                building_id_to_object[b].facility = facility
        Building.bulk_update(
            building_id_to_object.values(), fields=[Building.facility], batch_size=2000
        )

        new_building_sets_to_uuids = Facility._building_set_to_uuid()
        for building_set, uuid in old_building_sets_to_uuids.items():
            if building_set in new_building_sets_to_uuids:
                Facility.update(uuid=uuid).where(
                    Facility.uuid == new_building_sets_to_uuids[building_set]
                ).execute()
        # add uuids to facilities without them
        facilities_without_uuids = Facility.select().where(Facility.uuid.is_null())
        for f in tqdm(facilities_without_uuids):
            f.uuid = pw.fn.gen_random_uuid()
        # add uuid not null constraint
        Facility.bulk_update(
            facilities_without_uuids, fields=[Facility.uuid], batch_size=2000
        )
        Facility.raw("alter table facility alter column uuid set not null").execute()
        FacilityPermittedLocation.load()
        FacilityAnimalType.load()
        ConstructionAnnotation.load()
        from jobs import add_facility_field_to_permits

        add_facility_field_to_permits()
        # create index
        Facility.raw(
            "create index facility_geometry_idx on facility using gist(geometry)"
        ).execute()

    @classmethod
    def _add_uuids(cls):
        cls.raw("alter table facility add column uuid uuid").execute()
        cls.raw("update facility set uuid = gen_random_uuid()").execute()
        cls.raw("alter table facility alter column uuid set not null").execute()
        cls.raw(
            "alter table facility add constraint facility_uuid_unique unique(uuid)"
        ).execute()

    def to_gdf(self):
        gdf = gpd.GeoDataFrame.from_records(
            list(
                Building.select(
                    Building.id,
                    Building.geometry,
                    Building.latitude,
                    Building.longitude,
                )
                .where(Building.facility == self)
                .dicts()
            ),
        )
        gdf.crs = "EPSG:4326"
        return gdf

    def all_permits(self):
        return sum(
            [
                [
                    ppl.permit
                    for ppl in fpl.permitted_location.permit_permitted_locations
                ]
                for fpl in self.facility_permitted_locations
            ],
            [],
        )

    @property
    def animal_types(self):
        return [fat.animal_type for fat in self.facility_animal_types]

    def to_geojson_feature(self):
        geom = json.loads(shp.to_geojson(self.geometry))
        parcels = set([b.parcel for b in self.buildings])
        feature = {
            "type": "Feature",
            "geometry": geom,
            "id": str(self.uuid),
        }

        feature["properties"] = {
            "id": self.id,
            "uuid": str(self.uuid),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "lat_min": self.geometry.bounds[1],
            "lon_min": self.geometry.bounds[0],
            "lat_max": self.geometry.bounds[3],
            "lon_max": self.geometry.bounds[2],
            "parcels": [
                model_to_dict(p, exclude=[Parcel.data, Parcel.county])
                | {"county": p.county.name}
                for p in set([b.parcel for b in self.buildings])
                if p is not None
            ],
            "best_permits": [permit.data for permit in self.permits],
            "all_permits": [permit.data for permit in self.all_permits()],
            "construction_annotation": model_to_dict(
                self.construction_annotations[0],
                exclude=[ConstructionAnnotation.facility],
            ),
        }
        feature["bbox"] = [
            self.geometry.bounds[0],
            self.geometry.bounds[1],
            self.geometry.bounds[2],
            self.geometry.bounds[3],
        ]
        geom["bbox"] = feature["bbox"]
        return feature


class RegulatoryProgram(pw.Model):
    name = pw.TextField(unique=True)

    class Meta:
        database = db

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()
        with open("source_data/AFO_permits_all.csv") as f:
            permits = list(csv.DictReader(f))
        unique_programs = set()
        for permit in permits:
            for program in permit["Program"].split(","):
                unique_programs.add(program.strip())
        for program in unique_programs:
            if program and str(program) != "nan":
                cls.create(name=program)


class PermittedLocation(pw.Model):
    latitude = pw.DoubleField(null=True)
    longitude = pw.DoubleField(null=True)
    geometry = WktField()

    class Meta:
        database = db
        # unique index on the doublet
        indexes = ((("latitude", "longitude"), True),)

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()
        to_create = {}
        with open("source_data/AFO_permits_all.csv") as f:
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
        with open("source_data/afo_permits_geocoded.csv") as f:
            data = list(csv.DictReader(f))
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


class Parcel(pw.Model):
    numb = pw.TextField()
    county = pw.ForeignKeyField(County, backref="parcels")
    owner = pw.TextField()
    address = pw.TextField()
    data = JSONField()

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()

        parcel_dirs = (
            "source_data/polygon-centroids-merge-v2",
            "source_data/afo-permits-geocoded",
        )
        cls.add_from_directory(parcel_dirs[0])
        cls.add_from_directory(parcel_dirs[1])

    @classmethod
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

    @classmethod
    def load_from_dump(cls):
        cls.drop_table()
        cls.create_table()
        with open("outputs/parcels.csv") as f:
            reader = csv.DictReader(f)
            to_create = []
            for row in tqdm(reader):
                to_create.append(cls(**row))
            pbar = tqdm(total=len(to_create))
            while to_create:
                cls.bulk_create(to_create[:2000])
                to_create = to_create[2000:]
                pbar.update(2000)
            pbar.close()

    class Meta:
        indexes = ((("county_id", "numb"), True),)
        database = db


class Permit(pw.Model):
    wdid = pw.TextField()
    agency_name = pw.TextField()
    agency_address = pw.TextField()
    facility_name = pw.TextField()
    facility_address = pw.TextField()
    permitted_population = pw.IntegerField(null=True)
    regulatory_measure_status = pw.TextField(
        choices=[("Active", "active"), ("Historical", "historical")]
    )
    parcel = pw.ForeignKeyField(Parcel, backref="permits", null=True)
    facility = pw.ForeignKeyField(
        Facility,
        backref="permits",
        null=True,
    )
    data = JSONField()

    class Meta:
        database = db

    def all_locations(self):
        return [
            permit_permitted_location.permitted_location
            for permit_permitted_location in self.permit_permitted_locations
        ]

    def location(self):
        permitted_locations = {
            p.source: p.permitted_location for p in self.permit_permitted_locations
        }
        if "permit data" in permitted_locations:
            return permitted_locations["permit data"]
        return permitted_locations.get("address geocoding", None)

    def regulatory_programs(self):
        return [p.regulatory_program for p in self.permit_regulatory_programs]

    @classmethod
    def load(cls):
        with open("source_data/AFO_permits_all.csv") as f:
            data = list(csv.DictReader(f))

        PermitRegulatoryProgram.drop_table()
        PermitPermittedLocation.drop_table()
        cls.drop_table()
        cls.create_table()
        PermitPermittedLocation.create_table()
        PermitRegulatoryProgram.create_table()

        with open("source_data/afo_permits_geocoded.csv") as f:
            index_to_location = {
                int(row["permit_id"]): (float(row["lat"]), float(row["lon"]))
                for row in csv.DictReader(f)
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
            PermitPermittedLocation.bulk_create(
                permit_permitted_locations_to_create[:2000]
            )
            permit_permitted_locations_to_create = permit_permitted_locations_to_create[
                2000:
            ]
        while permit_regulatory_programs_to_create:
            PermitRegulatoryProgram.bulk_create(
                permit_regulatory_programs_to_create[:2000]
            )
            permit_regulatory_programs_to_create = permit_regulatory_programs_to_create[
                2000:
            ]
        from jobs import add_facility_field_to_permits, add_parcel_field_to_permits

        add_parcel_field_to_permits()
        add_facility_field_to_permits()


class FacilityPermittedLocation(pw.Model):
    facility = pw.ForeignKeyField(Facility, backref="facility_permitted_locations")
    permitted_location = pw.ForeignKeyField(
        PermittedLocation, backref="facility_permitted_locations"
    )
    distance = pw.DoubleField()

    class Meta:
        database = db

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()
        # connect every facility and permitted location 1km or less apart
        gdf = gpd.GeoDataFrame.from_records(
            list(Facility.select(Facility.id, Facility.geometry).dicts())
        )
        gdf.crs = "EPSG:4326"
        gdf = gdf.to_crs("EPSG:3311")
        facility_geometry_to_id = {f["geometry"]: f["id"] for i, f in gdf.iterrows()}
        gdf = gpd.GeoDataFrame.from_records(
            list(
                PermittedLocation.select(
                    PermittedLocation.id,
                    PermittedLocation.latitude,
                    PermittedLocation.longitude,
                ).dicts()
            )
        )
        gdf["geometry"] = gdf.apply(
            lambda row: shp.Point(row["longitude"], row["latitude"]), axis=1
        )
        gdf.crs = "EPSG:4326"
        gdf = gdf.to_crs("EPSG:3311")
        permitted_location_geometry_to_id = {
            f["geometry"]: f["id"] for i, f in gdf.iterrows()
        }
        permit_geoms = list(permitted_location_geometry_to_id.keys())
        index = shp.STRtree(permit_geoms)
        to_create = []
        for geom, i in tqdm(list(facility_geometry_to_id.items())):
            for j in index.query(geom, predicate="dwithin", distance=1000):
                to_create.append(
                    cls(
                        facility_id=i,
                        permitted_location_id=permitted_location_geometry_to_id[
                            permit_geoms[j]
                        ],
                        distance=geom.distance(permit_geoms[j]),
                    )
                )
        while to_create:
            cls.bulk_create(to_create[:2000])
            to_create = to_create[2000:]


class PermitPermittedLocation(pw.Model):
    permit = pw.ForeignKeyField(Permit, backref="permit_permitted_locations")
    permitted_location = pw.ForeignKeyField(
        PermittedLocation, backref="permit_permitted_locations"
    )
    source = pw.TextField(
        choices=[
            ("permit data", "permit data"),
            ("address geocoding", "address geocoding"),
        ]
    )

    class Meta:
        database = db

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()


class PermitRegulatoryProgram(pw.Model):
    permit = pw.ForeignKeyField(Permit, backref="permit_regulatory_programs")
    regulatory_program = pw.ForeignKeyField(
        RegulatoryProgram, backref="permit_regulatory_programs"
    )

    class Meta:
        database = db

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()


class Image(pw.Model):
    name = pw.TextField(unique=True)
    county = pw.ForeignKeyField(County, backref="images")
    lon_min = pw.DoubleField()
    lon_max = pw.DoubleField()
    lat_min = pw.DoubleField()
    lat_max = pw.DoubleField()
    geometry = WktField()

    bucket = pw.CharField(
        choices=[
            ("0", "0"),
            ("1", "1"),
            ("1.25", "1.25"),
            ("1.75", "1.75"),
            ("3", "3"),
            ("inf", "inf"),
            ("ex ante permit", "ex ante permit"),
            ("completed", "completed"),
        ],
        null=True,
    )

    label_status = pw.CharField(
        choices=[
            ("active learner", "active learner"),
            ("post hoc permit", "post hoc permit"),
            ("adjacent", "adjacent"),
            ("unlabeled", "unlabeled"),
            ("removed", "removed"),
        ],
        null=True,
    )

    stratum = pw.CharField(null=True)

    class Meta:
        database = db

    @classmethod
    def _geom_tree(cls):
        if not hasattr(cls, "_cache_geom_tree"):
            geoms = cls.select(cls.name, cls.geometry).dicts()
            cls._cache_geom_tree = shp.STRtree([g["geometry"] for g in geoms])
            cls._cache_geom_names = [g["name"] for g in geoms]
        return cls._cache_geom_tree, cls._cache_geom_names

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()

        csv.field_size_limit(sys.maxsize)
        county_name_to_id = {
            c["name"]: c["id"] for c in County.select(County.name, County.id).dicts()
        }

        to_create = []
        for fname in tqdm(os.listdir("source_data/image_locations")):
            with open(os.path.join("source_data/image_locations", fname)) as f:
                images = list(csv.DictReader(f))
            for image in images:
                county = county_name_to_id[image["county"]]
                im = cls(
                    name=image["name"].split(".")[0],
                    county=county,
                    lon_min=image["lon_min"],
                    lon_max=image["lon_max"],
                    lat_min=image["lat_min"],
                    lat_max=image["lat_max"],
                    geometry=shp.from_wkt(
                        shp.Polygon.from_bounds(
                            float(image["lon_min"]),
                            float(image["lat_min"]),
                            float(image["lon_max"]),
                            float(image["lat_max"]),
                        ).wkt
                    ),
                )
                to_create.append(im)
        while to_create:
            cls.bulk_create(to_create[:2000])
            to_create = to_create[2000:]

        # create index
        cls.raw(
            "create index image_geometry_idx on image using gist(geometry)"
        ).execute()

    @classmethod
    def get_images_for_area(cls, lon_min, lat_min, lon_max, lat_max):
        geom = shp.Polygon.from_bounds(lon_min, lat_min, lon_max, lat_max)
        # use postgis index to get images that intersect the area
        images = cls.raw(
            "select * from image where ST_INTERSECTS(geometry, '{}')".format(
                geom.wkt,
            ),
        )
        return list(images)

    def facilities(self):
        return list(
            {building.facility for building in self.buildings if building.facility}
        )

    def adjacent_images(self):
        return list(
            Image.select().where(
                Image.id.in_(
                    self._image_adjacency.select(ImageAdjacency.adjacent_image_id)
                )
            )
        )


class ImageAdjacency(pw.Model):
    image = pw.ForeignKeyField(Image, backref="_image_adjacency")
    adjacent_image = pw.ForeignKeyField(Image, backref="_image_adjacency_adjacent")

    class Meta:
        database = db
        indexes = ((("image_id", "adjacent_image_id"), True),)

    @classmethod
    def load(cls):
        images = list(Image.select(Image.id, Image.geometry).dicts())
        complete = {
            i[0] for i in ImageAdjacency.select(ImageAdjacency.image_id).tuples()
        }
        geom_tree = shp.STRtree([i["geometry"] for i in images])
        to_create = []
        for i, image in tqdm(enumerate(images)):
            if i in complete:
                continue
            intersections = geom_tree.query(image["geometry"], predicate="intersects")
            for j in intersections:
                if i == j:
                    continue
                to_create.append(
                    cls(
                        image_id=images[i]["id"],
                        adjacent_image_id=images[j]["id"],
                    )
                )
        import ipdb

        ipdb.set_trace()
        while to_create:
            cls.bulk_create(to_create[:20000])
            to_create = to_create[20000:]


class Building(pw.Model):
    latitude = pw.DoubleField()
    longitude = pw.DoubleField()
    area_sqm = pw.DoubleField()
    image_xy_poly = WktField(null=True)
    geometry = WktField(unique=True)
    legacy_index = pw.IntegerField(null=True)
    parcel = pw.ForeignKeyField(Parcel, backref="buildings", null=True)
    facility = pw.ForeignKeyField(Facility, backref="buildings", null=True)
    image = pw.ForeignKeyField(Image, backref="buildings", null=True)
    cafo = pw.BooleanField(null=True)

    class Meta:
        database = db
        indexes = ((("latitude", "longitude"), True),)

    @staticmethod
    def _loc_to_parcel(lat, lon):
        parcels_path = "source_data/polygon-centroids-merge-v2"
        try:
            return Building._loc_to_parcel_mapping.get((lat, lon), None)
        except AttributeError:
            pass
        Building._loc_to_parcel_mapping = {}
        parcel_ids = {
            (p["county"], p["numb"]): p["id"] for p in Parcel.select().dicts()
        }
        for fname in os.listdir(parcels_path):
            with open(os.path.join(parcels_path, fname)) as f:
                rows = list(csv.DictReader(f))
            county = (
                County.select()
                .where(
                    County.name
                    == " ".join(fname.split(".")[0].split("-")[3:-2]).title()
                )
                .first()
            )
            for r in rows:
                Building._loc_to_parcel_mapping[
                    (float(r["Latitude"]), float(r["Longitude"]))
                ] = parcel_ids.get((county.id, r["parcelnumb"]), None)

        return Building._loc_to_parcel_mapping.get((lat, lon), None)

    @classmethod
    def load(cls):
        import geopandas as gpd

        cls.drop_table()
        cls.create_table()
        gdf = gpd.read_file("source_data/building_polygons")

        for i, row in gdf.iterrows():
            try:
                image_xy_poly = shp.from_wkt(row["poly_xy"])
            except shp.GEOSException:
                image_xy_poly = None
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                cls.create(
                    latitude=lat,
                    longitude=lon,
                    image=Image.select()
                    .where(Image.name == row["image_name"].split(".")[0])
                    .first(),
                    image_xy_poly=image_xy_poly,
                    area_sqm=row["area_sqm"],
                    geometry=row["geometry"],
                    legacy_index=i,
                    parcel_id=Building._loc_to_parcel(lat, lon),
                )
            except pw.IntegrityError as e:
                duplicate = (
                    cls.select()
                    .where((cls.latitude == lat) & (cls.longitude == lon))
                    .first()
                )
                if not duplicate:
                    print(f"Failed to load building {i} at {lat}, {lon} due to {e}")
                elif shp.difference(duplicate.geometry, row["geometry"]).area == 0:
                    print("Skipping duplicate building")
                else:
                    raise e
        # create index
        cls.raw(
            "create index building_geometry_idx on building using gist(geometry)"
        ).execute()

    @classmethod
    def load_from_dump(cls):
        cls.drop_table()
        cls.create_table()
        cls.raw("alter table building alter column id drop default").execute()
        with open("outputs/buildings.csv") as f:
            reader = csv.DictReader(f)
            to_create = []
            created = set()
            for row in reader:
                if (row["latitude"], row["longitude"]) in created:
                    print("Duplicate building")
                    print(row)
                    continue
                row = row | {key: None for key in row if not row[key]}
                row["geometry"] = shp.from_wkt(row["geometry"])
                row["image_xy_poly"] = shp.from_wkt(row["image_xy_poly"])
                created.add((row["latitude"], row["longitude"]))
                to_create.append(row)

            pbar = tqdm(total=len(to_create))
            while to_create:
                # peewee.bulk_create ignores id fields, so we have to do this
                cls.insert_many(to_create[:2000]).execute()
                to_create = to_create[2000:]
                pbar.update(2000)
            pbar.close()

    @classmethod
    def ingest_from_json_directory(cls, directory):
        bulk = []
        json_files = glob.glob(f"{directory}/*.json")
        pbar = tqdm(total=len(json_files))

        def ingest_json(json_file):
            with open(json_file) as f:
                data = json.load(f)
            b = cls.construct_from_annotation_dict(data)
            bulk.extend(b)
            pbar.update(1)

        def ingest_set(json_files):
            for json_file in json_files:
                ingest_json(json_file)

        threads = 16
        with ThreadPool(threads) as pool:
            pool.map(ingest_set, [json_files[i::threads] for i in range(threads)])
        cls.bulk_create(bulk)

    @classmethod
    def _get_image_with_name(cls, name):
        if not hasattr(cls, "_image_names"):
            cls._image_names = {i.name: i for i in Image.select(Image.name, Image.id)}
        return cls._image_names[name]

    @classmethod
    def construct_from_annotation_dict(cls, data):
        if (len(data["annotations"]) == 1) and (
            data["annotations"][0]["label"] == "Blank"
        ):
            return []
        image_name = data["name"].split(".")[0].strip("/")
        image = cls._get_image_with_name(image_name)
        legacy_index = None
        parcel = None
        facility = None
        raster_dataset = rasterio.open(
            naip.download_ca_cafo_naip_image(image_name, naip.Format.TIF)
        )
        to_3311 = lambda x, y: rasterio.warp.transform("EPSG:4326", "EPSG:3311", x, y)

        bulk = []
        for annotation in data["annotations"]:
            if annotation["label"] != "cafo" or annotation["type"] != "segment":
                continue
            if "coordinates" not in annotation:
                print(f"Skipping annotation without coordinates in {image_name}")
                continue
            if len(annotation["coordinates"]) != 1:
                raise ValueError(
                    f"Unexpected number of coordinates in {image_name}: {len(annotation['coordinates'])}"
                )
            pixels = [(a["x"], a["y"]) for a in annotation["coordinates"][0]]
            image_xy_poly = shp.Polygon(pixels)
            coords = [
                raster_dataset.xy(pixel[1], pixel[0], offset="ul") for pixel in pixels
            ]
            geometry = shp.Polygon(coords)
            lon, lat = geometry.centroid.coords[0]
            area_sqm = shp.ops.transform(to_3311, geometry).area
            bulk.append(
                cls(
                    latitude=lat,
                    longitude=lon,
                    image=image,
                    image_xy_poly=image_xy_poly,
                    area_sqm=area_sqm,
                    geometry=geometry,
                    legacy_index=legacy_index,
                    parcel=parcel,
                    facility=facility,
                )
            )
        return bulk


class BuildingRelationship(pw.Model):
    building = pw.ForeignKeyField(Building, backref="building_relationships")
    other_building = pw.ForeignKeyField(Building)
    reason = pw.TextField(
        choices=(
            ("matching parcel", "matching parcel"),
            ("distance", "distance"),
            ("parcel name tf-idf", "parcel name tf-idf"),
            ("parcel name fuzzy", "parcel name fuzzy"),
            ("parcel name annotation", "parcel name annotation"),
            ("override: unite", "override: unite"),
            ("override: separate", "override: separate"),
        )
    )
    weight = pw.IntegerField(null=True)

    class Meta:
        database = db
        indexes = ((("building_id", "other_building_id", "reason"), True),)
        constraints = [
            pw.Check("building_id != other_building_id"),
        ]

    @classmethod
    def create_pair(cls, building_1, building_2, reason, weight=None):
        cls.create(
            building=building_1,
            other_building=building_2,
            reason=reason,
            weight=weight,
        )
        cls.create(
            building=building_2,
            other_building=building_1,
            reason=reason,
            weight=weight,
        )

    @classmethod
    def construct_set(cls, building_set, reason, weight=None):
        return [
            cls(
                building=building_1,
                other_building=building_2,
                reason=reason,
                weight=weight,
            )
            for building_1, building_2 in itertools.permutations(building_set, 2)
        ]

    @classmethod
    def add_parcel_name_annotations(cls):
        cls.delete().where(cls.reason == "parcel name annotation").execute()
        with open("source_data/parcel_name_annotations.csv") as f:
            parcel_owner_annotations = list(csv.DictReader(f))
        bulk = []
        buildings = Building.select()
        parcels = Parcel.select()
        buildings = pw.prefetch(buildings, parcels)

        owners_to_buildings = {}
        for b in buildings:
            owner = None
            if b.parcel:
                owner = b.parcel.owner
            if owner:
                owners_to_buildings[owner] = owners_to_buildings.get(owner, []) + [b]
        buildings_with_adjacencies = set(
            BuildingRelationship.select(
                BuildingRelationship.building_id, BuildingRelationship.other_building_id
            )
            .tuples()
            .distinct()
        )

        bulk = []

        for row in tqdm(parcel_owner_annotations):
            if not int(row["create_override_match"]):
                continue
            buildings_1 = owners_to_buildings.get(row["owner_1"], [])
            buildings_2 = owners_to_buildings.get(row["owner_2"], [])
            for b in buildings_1:
                for b2 in buildings_2:
                    adjacency = (b.id, b2.id) in buildings_with_adjacencies
                    if adjacency:
                        bulk.append(
                            cls(
                                building=b,
                                other_building=b2,
                                reason="parcel name annotation",
                                weight=1000,
                            )
                        )
                        bulk.append(
                            cls(
                                building=b2,
                                other_building=b,
                                reason="parcel name annotation",
                                weight=1000,
                            )
                        )
        while bulk:
            cls.bulk_create(bulk[:2000])
            bulk = bulk[2000:]

    @classmethod
    def add_parcel_matches(cls):
        buildings = (
            Building.select(Building.id, Building.parcel)
            .where(Building.parcel.is_null(False))
            .order_by(Building.parcel)
            .dicts()
        )
        idx = 0
        prev_idx = 0
        bulk = []
        pbar = tqdm(total=len(buildings))
        while idx < len(buildings):
            parcel = buildings[idx]["parcel"]
            while idx < len(buildings) and buildings[idx]["parcel"] == parcel:
                idx += 1
                pbar.update(1)
            if idx - prev_idx > 1:
                bulk += cls.construct_set(
                    [b["id"] for b in buildings[prev_idx:idx]], "matching parcel"
                )
            prev_idx = idx
            while bulk:
                cls.bulk_create(bulk[:2000])
                bulk = bulk[2000:]
        pbar.close()

    @classmethod
    def add_parcel_name_tfidf_matches(cls):
        distance_matches = (
            cls.select(cls.building, cls.other_building)
            .where(cls.reason == "distance")
            .count()
        )
        if not distance_matches:
            raise ValueError(
                "No adjacency matches; adjacency matches must be added first"
            )
        documents = Parcel.select(Parcel.id, Parcel.owner).tuples()
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([d[1] for d in documents])
        documents = {d[0]: t for d, t in zip(documents, tfidf.toarray())}
        bulk = []
        existing_parcel_name_tfidf_matches = cls.select(
            cls.building_id, cls.other_building_id
        ).where((cls.reason == "parcel name tf-idf"))
        distance_matches = (
            cls.select(cls.building, cls.other_building)
            .where((cls.reason == "distance"))
            .join(
                existing_parcel_name_tfidf_matches,
                on=(
                    (
                        cls.building_id
                        == existing_parcel_name_tfidf_matches.c.building_id
                    )
                    & (
                        cls.other_building_id
                        == existing_parcel_name_tfidf_matches.c.other_building_id
                    )
                ),
                join_type=pw.JOIN.LEFT_OUTER,
            )
            .where(existing_parcel_name_tfidf_matches.c.building_id.is_null())
            .dicts()
        )
        building_parcel_mapping = {
            b.id: b.parcel_id
            for b in Building.select(Building.id, Building.parcel_id)
            if b.parcel_id
        }
        for match in tqdm(distance_matches):
            parcel = building_parcel_mapping.get(match["building"], None)
            other_parcel = building_parcel_mapping.get(match["other_building"], None)
            if not parcel or not other_parcel:
                continue
            document = documents[parcel]
            other_document = documents[other_parcel]
            similarity = int(document.dot(other_document.T) * 1000)
            bulk.append(
                cls(
                    building=match["building"],
                    other_building=match["other_building"],
                    reason="parcel name tf-idf",
                    weight=similarity,
                )
            )
        pb = tqdm(total=len(bulk))
        while bulk:
            cls.bulk_create(bulk[:20000])
            bulk = bulk[20000:]
            pb.update(20000)

    FUZZY_MATCH_WORDS_TO_REMOVE = [
        "llc",
        "farms",
        "trust",
        "tr",
        "trustee",
        "dairy",
        "inc",
        "revocable",
        "irrevocable",
        "farm",
        "family",
        "poultry",
        "cattle",
        "ranch",
        "acres",
        "land",
        "real",
        "estate",
        "ridge",
        "john",
        "fam",
        "partnership",
        "prop",
        "enterprises",
        "landowner",
        "lp",
    ]

    @staticmethod
    def fuzzy_name_match(name_1, name_2):
        from thefuzz import fuzz

        names = [name_1, name_2]
        name_1 = re.sub(r"[^\w\s]", " ", name_1.lower())
        name_2 = re.sub(r"[^\w\s]", " ", name_2.lower())
        for word in BuildingRelationship.FUZZY_MATCH_WORDS_TO_REMOVE:
            name_1 = name_1.replace(word, "")
            name_2 = name_2.replace(word, "")
        name_1 = " ".join(name_1.split())
        name_2 = " ".join(name_2.split())
        # scale to 1000
        return fuzz.ratio(name_1, name_2) * 10

    @classmethod
    def add_parcel_name_fuzzy_matches(cls):
        distance_matches = (
            cls.select(cls.building, cls.other_building)
            .where(cls.reason == "distance")
            .count()
        )
        cls.delete().where(cls.reason == "parcel name fuzzy").execute()
        if not distance_matches:
            raise ValueError(
                "No adjacency matches; adjacency matches must be added first"
            )
        bulk = []
        existing_parcel_name_fuzzy_matches = cls.select(
            cls.building_id, cls.other_building_id
        ).where((cls.reason == "parcel name fuzzy"))
        distance_matches = (
            cls.select(cls.building, cls.other_building)
            .join(
                existing_parcel_name_fuzzy_matches,
                on=(
                    (
                        cls.building_id
                        == existing_parcel_name_fuzzy_matches.c.building_id
                    )
                    & (
                        cls.other_building_id
                        == existing_parcel_name_fuzzy_matches.c.other_building_id
                    )
                ),
                join_type=pw.JOIN.LEFT_OUTER,
            )
            .where(
                (cls.reason == "distance")
                & (existing_parcel_name_fuzzy_matches.c.building_id.is_null())
            )
            .dicts()
        )
        building_owner_mapping = {
            b["id"]: b["owner"]
            for b in Building.select(
                Building.id.alias("id"), Parcel.owner.alias("owner")
            )
            .join(Parcel)
            .dicts()
        }
        for match in tqdm(distance_matches):
            owner = building_owner_mapping.get(match["building"], None)
            other_owner = building_owner_mapping.get(match["other_building"], None)
            if not owner or not other_owner:
                continue
            similarity = BuildingRelationship.fuzzy_name_match(owner, other_owner)
            bulk.append(
                cls(
                    building=match["building"],
                    other_building=match["other_building"],
                    reason="parcel name fuzzy",
                    weight=similarity,
                )
            )
        pb = tqdm(total=len(bulk))
        while bulk:
            cls.bulk_create(bulk[:20000])
            bulk = bulk[20000:]
            pb.update(20000)

    @classmethod
    def add_distance_matches(cls):
        import geopandas as gpd
        import numpy as np
        import scipy as sp
        import shapely as shp

        BuildingRelationship.delete().where(cls.reason == "distance").execute()
        geometries = gpd.GeoSeries(
            [
                geom[0]
                for geom in Building.select(Building.geometry)
                .order_by(Building.id)
                .tuples()
            ]
        )

        ids = sum(Building.select(Building.id).order_by(Building.id).tuples(), tuple())
        geometries.crs = "EPSG:4326"
        # cartesian projections in meters
        geometries = geometries.to_crs("EPSG:3311")
        tree = shp.STRtree(geometries)
        input_idxs, tree_idxs = tree.query(
            geometries, predicate="dwithin", distance=1000
        )
        distances = np.array(
            [
                geometries[i].distance(geometries[j])
                for i, j in zip(input_idxs, tree_idxs)
            ]
        )

        bulk = []
        for input_idx, tree_idx, distance in tqdm(
            zip(input_idxs, tree_idxs, distances), total=len(ids)
        ):
            if input_idx == tree_idx:
                continue
            bulk.append(
                cls(
                    building=ids[input_idx],
                    other_building=ids[tree_idx],
                    reason="distance",
                    weight=1000 - int(distance),
                )
            )

        pb = tqdm(total=len(bulk))
        while bulk:
            cls.bulk_create(bulk[:20000])
            bulk = bulk[20000:]
            pb.update(20000)
        pb.close()

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()
        cls.add_parcel_matches()
        cls.add_distance_matches()
        cls.add_parcel_name_tfidf_matches()
        cls.add_parcel_name_fuzzy_matches()
        cls.add_parcel_name_annotations()

    @classmethod
    def load_from_dump(cls):
        cls.drop_table()
        cls.create_table()
        building_ids = set(sum(Building.select(Building.id).tuples(), tuple()))
        missing_buildings = [
            n for n in range(1, max(building_ids) + 1) if n not in building_ids
        ]
        if len(missing_buildings) > 3:
            raise ValueError(f"Missing more than 3 buildings: {missing_buildings}...")
        with open("outputs/building_relationships.csv") as f:
            reader = csv.DictReader(f)
            to_create = []
            for row in tqdm(reader):
                if (
                    int(row["building_id"]) in missing_buildings
                    or int(row["other_building_id"]) in missing_buildings
                ):
                    continue
                if (
                    int(row["building_id"]) not in building_ids
                    or int(row["other_building_id"]) not in building_ids
                ):
                    print(f"Warning: skipping relationship for invalid building {row}")
                    continue
                to_create.append(cls(**row | {"weight": row["weight"] or None}))
            pbar = tqdm(total=len(to_create))
            while to_create:
                cls.bulk_create(to_create[:2000])
                to_create = to_create[2000:]
                pbar.update(2000)
            pbar.close()

    @classmethod
    def retrieve_thresholded_building_relationships(
        cls,
        distance=400,
        tfidf=700,
        fuzzy=600,
        fuzzy_max=1001,
        tfidf_max=1001,
        no_owner_distance=200,
        lone_building_distance=50,
    ):
        return cls.raw(
            BUILDING_THRESHOLD_RELATIONSHIP_QUERY.format(
                distance=distance,
                tfidf=tfidf,
                fuzzy=fuzzy,
                fuzzy_max=fuzzy_max,
                tfidf_max=tfidf_max,
                no_owner_distance=no_owner_distance,
                lone_building_distance=lone_building_distance,
            )
        )

    @classmethod
    def dict_of_lists(cls, **kwargs):
        dol = {
            b[0]: set()
            for b in Building.select(Building.id).where(Building.cafo).tuples()
        }
        query = cls.retrieve_thresholded_building_relationships(**kwargs)
        for br in query.dicts():
            try:
                dol[br["building"]].add(br["other_building"])
            except KeyError:
                pass
        return dol

    @classmethod
    def graph(cls, **kwargs):
        import networkx as nx

        return nx.from_dict_of_lists(cls.dict_of_lists(**kwargs))

    @classmethod
    def facilities(cls, **kwargs):
        import networkx as nx

        return nx.connected_components(cls.graph(**kwargs))


class AnimalType(pw.Model):
    name = pw.TextField(unique=True)

    class Meta:
        database = db

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()
        for name in (
            "cows",
            "pigs",
            "poultry",
            "dairy",
            "goats",
            "sheep",
            "auction",
            "calves",
            "horses",
            "not afo",
        ):
            cls.create(name=name)


class FacilityAnimalType(pw.Model):
    facility = pw.ForeignKeyField(Facility, backref="facility_animal_types")
    animal_type = pw.ForeignKeyField(AnimalType, backref="facility_animal_types")
    label_source = pw.TextField(choices=[("human", "human"), ("permit", "permit")])

    class Meta:
        database = db
        indexes = ((("facility", "animal_type"), True),)

    @staticmethod
    @cache
    def legacy_facility_id_to_facility():
        with open("source_data/8-10-23_facility_centroids_all.csv") as f:
            legacy_facility_id_to_poly_ids = {
                int(float(row["facility_id"])): eval(row["poly_id"])
                for row in csv.DictReader(f)
            }
        legacy_index_to_facility = {
            row["legacy_index"]: row["id"]
            for row in Facility.select(Facility.id, Building.legacy_index)
            .join(Building)
            .dicts()
        }
        mapping = {}
        for fid, poly_ids in legacy_facility_id_to_poly_ids.items():
            for poly_id in poly_ids:
                if poly_id in legacy_index_to_facility:
                    mapping[fid] = mapping.get(fid, set()) | {
                        legacy_index_to_facility[poly_id]
                    }
        return mapping

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()

        with open("source_data/facilities_with_animal_types.csv") as f:
            data = list(csv.DictReader(f))
        animal_types = {row["name"]: row["id"] for row in AnimalType.select().dicts()}
        mapping = cls.legacy_facility_id_to_facility()
        to_create = []
        legacy_ids_covered = set()
        for row in data:
            legacy_facility_id = int(float(row["facility_i"]))
            legacy_ids_covered.add(legacy_facility_id)
            if legacy_facility_id in (2217, 2218):
                continue
            try:
                facility_ids = mapping[legacy_facility_id]
            except KeyError:
                print(f"Missing facility for legacy id {legacy_facility_id}")
                continue
            for facility_id in facility_ids:
                types = 0
                for animal_type, at_id in animal_types.items():
                    if animal_type == "poultry":
                        animal_type = "chickens"
                    if str(row.get(animal_type, "")) == "True":
                        types += 1
                        to_create.append(
                            cls(
                                facility_id=facility_id,
                                animal_type_id=at_id,
                                label_source="human"
                                if "permit" not in row["labeler"].lower()
                                else "permit",
                            )
                        )
                if types == 0:
                    print("Missing animal type for facility", legacy_facility_id)
        missed = set(mapping.keys()) - legacy_ids_covered
        if missed:
            print(f"Missed {len(missed)} legacy ids", missed)
            missed_facilities = set.union(*[mapping[fid] for fid in missed])
            print(f"Missed {len(missed_facilities)} facilities", missed_facilities)
        # remove duplicates
        to_create = list(
            {(c.facility_id, c.animal_type_id): c for c in to_create}.values()
        )
        while to_create:
            cls.bulk_create(to_create[:200])
            to_create = to_create[200:]


class ConstructionAnnotation(pw.Model):
    facility = pw.ForeignKeyField(
        Facility, backref="construction_annotations", unique=True, null=True
    )
    construction_lower_bound = pw.IntegerField(null=True)
    construction_upper_bound = pw.IntegerField()
    destruction_lower_bound = pw.IntegerField(null=True)
    destruction_upper_bound = pw.IntegerField(null=True)
    significant_population_change = pw.BooleanField()
    indoor_outdoor = pw.TextField(
        choices=[("indoor", "indoor"), ("outdoor", "outdoor")]
    )
    has_lagoon = pw.BooleanField()

    @staticmethod
    def _get_facilities_for_annotation_dict(annotation_dict):
        mapping = FacilityAnimalType.legacy_facility_id_to_facility()
        uuid_to_cafo_id = ConstructionAnnotation._uuid_to_cafo_id_mapping()
        if "CAFO UUID" in annotation_dict:
            lat = float(
                annotation_dict.get("Latitude")
                or annotation_dict.get("lat,lon").split(",")[0]
            )
            lon = float(
                annotation_dict.get("Longitude")
                or annotation_dict.get("lat,lon").split(",")[1]
            )
            try:
                facility_ids = uuid_to_cafo_id[annotation_dict["CAFO UUID"]]
            except KeyError:
                facility_ids = sum(
                    list(
                        Facility.select(Facility.id)
                        .where(
                            pw.fn.ST_Contains(
                                pw.fn.ST_Envelope(Facility.geometry),
                                pw.fn.ST_Point(lon, lat),
                            )
                        )
                        .tuples()
                    ),
                    (),
                )
            if not facility_ids:
                print(f"Missing facility for uuid {annotation_dict['CAFO UUID']}")
        else:
            annotation_dict["CAFO ID"] = int(annotation_dict["CAFO ID"])
            if annotation_dict["CAFO ID"] not in mapping:
                print(f"Missing facility for legacy id {annotation_dict['CAFO ID']}")
                return []
            facility_ids = mapping[annotation_dict["CAFO ID"]]
        return facility_ids

    @staticmethod
    def _merge_annotation_dicts(annotation_dicts):
        merged = {}
        for annotation_dict in annotation_dicts:
            for key in [
                "construction lower",
                "construction upper",
                "destruction lower",
                "destruction upper",
                "significant animal population change",
                "where animals stay",
                "facility has a lagoon",
            ]:
                value = annotation_dict[key]
                if key not in merged:
                    merged[key] = value
                elif key in (
                    "significant animal population change",
                    "facility has a lagoon",
                ):
                    merged[key] = merged[key] or value
                elif merged[key] != value:
                    raise ValueError(
                        f"Conflicting values for {key}: {merged[key]} and {value}"
                    )
        return merged

    @staticmethod
    @cache
    def _uuid_to_cafo_id_mapping():
        return {
            facility["uuid"]: facility["id"]
            for facility in Facility.select(Facility.id, Facility.uuid).dicts()
        }

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()
        rows = []
        for file_ in [
            "construction_dating_v1.csv",
            "construction_dating_adjacent_images_v1.csv",
            "construction_dating_ex_ante_permits.csv",
            "construction_dating_ex_ante_adjacents.csv",
            "construction_dating_missed_adjacents.csv",
        ]:
            with open(data.get(file_)) as f:
                rows += list(csv.DictReader(f))
        to_create = {}

        facilities_to_annotations = {}
        non_cafo_buildings = set()

        for row in rows:
            row = {key.replace("\n", " "): row[key] for key in row}
            if row["CAFO ID"] == "<sample>":
                continue
            facility_ids = cls._get_facilities_for_annotation_dict(row)
            if row.get("is cafo") == "False":
                buildings = sum(
                    Building.select(Building.id)
                    .where(Building.facility_id << facility_ids)
                    .tuples(),
                    (),
                )
                non_cafo_buildings |= set(buildings)

            if row["construction upper"].strip("-") == "":
                if row["CF notes"] or row["Cloudfactory review feedback"]:
                    continue
                print("Missing construction upper bound for CAFO", row["CAFO ID"])
                continue
            for facility_id in facility_ids:
                facilities_to_annotations[facility_id] = facilities_to_annotations.get(
                    facility_id, []
                ) + [row]

        for facility_id, rows in facilities_to_annotations.items():
            for row in rows:
                row["Review Date"] = row["Review Date"] or "11/9/2023"
            rows = sorted(
                rows,
                key=lambda r: datetime.strptime(r["Review Date"], "%m/%d/%Y"),
                reverse=True,
            )
            rows = [r for r in rows if r["Review Date"] == rows[0]["Review Date"]]
            try:
                row = cls._merge_annotation_dicts(rows)
            except ValueError as e:
                print(f"Conflicting annotations for facility {facility_id}", e)
                continue
            if not row["construction upper"].strip("-"):
                print(
                    f"Missing construction upper bound for facility {facility_id}, legacy id {row['CAFO ID']}"
                )
                continue

            obj = cls(
                facility_id=facility_id,
                construction_lower_bound=row["construction lower"].strip("-") or None,
                construction_upper_bound=row["construction upper"],
                destruction_lower_bound=row["destruction lower"].strip("-") or None,
                destruction_upper_bound=row["destruction upper"].strip("-") or None,
                significant_population_change=row[
                    "significant animal population change"
                ],
                indoor_outdoor=row["where animals stay"],
                has_lagoon=(row["facility has a lagoon"] == "True"),
            )
            to_create[facility_id] = obj

        to_create = list(to_create.values())
        while to_create:
            cls.bulk_create(to_create[:2000])
            to_create = to_create[2000:]

        Building.update(facility_id=None, cafo=False).where(
            Building.id << non_cafo_buildings
        ).execute()

    class Meta:
        database = db


class RawCloudFactoryImageAnnotation(pw.Model):
    image = pw.ForeignKeyField(
        Image, backref="raw_cloud_factory_image_annotations", unique=True
    )
    created_at = pw.DateTimeField()
    json = JSONField()

    @classmethod
    def dump(cls, output_jsonl):
        with open(output_jsonl, "w") as f:
            for r in cls.select().dicts():
                r["created_at"] = r["created_at"].isoformat()
                f.write(json.dumps(r) + "\n")

    class Meta:
        database = db


def models():
    return [model for model in pw.Model.__subclasses__() if model._meta.database]


def dependencies():
    deps = {}
    for model in models():
        deps[model] = set()
        for field in model._meta.fields.values():
            if not isinstance(field, pw.ForeignKeyField):
                continue
            deps[model].add(field.rel_model)
    return deps


def _drop_all(verbose=True):
    deps = dependencies()
    remaining = set(models())
    while remaining:
        independent = remaining - set.union(*deps.values())
        if not independent:
            raise ValueError(
                f"Circular or deadlocked dependencies present in dependency graph {deps}"
            )
        for model in independent:
            if verbose:
                print(f"Dropping {model}")
            model.drop_table()
        remaining -= independent
        for model in independent:
            del deps[model]


def _load_all(verbose=True):
    deps = dependencies()
    remaining = set(models())
    while remaining:
        independent = {r for r in remaining if not deps[r]}
        if not independent:
            raise ValueError(
                f"Circular or deadlocked dependencies present in dependency graph {deps}"
            )
        for model in independent:
            if verbose:
                print(f"Loading {model}")
            if hasattr(model, "load_from_dump"):
                model.load_from_dump()
            else:
                model.load()
        remaining -= independent
        for d in deps.values():
            d -= independent


def reload_all(warn=True, custom_db=None):
    if warn:
        con = input(
            "Warning: you are about to drop all data and attempt to recreate it from sources. If sources or missing or if you have made custom modifications, they will be lost. Continue? (Y/n)"
        )
        if con.lower() not in ("y", "yes"):
            print("exiting.")
            return

    _drop_all()
    _load_all()


def make_spatial_indexes():
    for model in models():
        if hasattr(model, "geometry"):
            model.raw(
                f"create index {model._meta.table_name}_geometry_idx on {model._meta.table_name} using gist(geometry)"
            ).execute()


if __name__ == "__main__":
    pass
