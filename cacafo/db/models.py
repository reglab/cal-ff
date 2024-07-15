import json

import geopandas as gpd
import peewee as pw
import shapely as shp
from playhouse.postgres_ext import JSONField, PostgresqlExtDatabase
from playhouse.shortcuts import model_to_dict

from cacafo.db.session import get_peewee_connection

db = get_peewee_connection()


class WktField(pw.Field):
    field_type = "text"

    def db_value(self, value):
        if value is None:
            return value
        if isinstance(value, shp.Geometry):
            return value.wkt
        raise TypeError("A WktField requires shapely.lib.Geometry")

    def python_value(self, value):
        if value is None:
            return None
        return shp.from_wkt(value)


class CountyGroup(pw.Model):
    name = pw.TextField(unique=True)

    class Meta:
        database = db


class County(pw.Model):
    name = pw.TextField(unique=True)
    latitude = pw.DoubleField()
    longitude = pw.DoubleField()
    geometry = WktField()
    county_group = pw.ForeignKeyField(CountyGroup, backref="counties", null=True)

    class Meta:
        database = db

    @classmethod
    def geocode(cls, lon=None, lat=None):
        if None in (lon, lat):
            raise ValueError("None values given to goecode")
        point = shp.Point(lon, lat)
        for county in cls._counties():
            if county.geometry.contains(point):
                return county
        raise ValueError(f"No county found for {point}")

    @classmethod
    def _counties(cls):
        if not hasattr(cls, "_cache_counties"):
            cls._cache_counties = list(cls.select())
        return cls._cache_counties


class Facility(pw.Model):
    latitude = pw.DoubleField()
    longitude = pw.DoubleField()
    geometry = WktField()
    uuid = pw.UUIDField(unique=True)

    class Meta:
        database = db

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
        gdf = gdf.set_geometry("geometry")
        gdf.crs = "EPSG:4326"
        return gdf

    @property
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
            "all_permits": [permit.data for permit in self.all_permits],
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


class PermittedLocation(pw.Model):
    latitude = pw.DoubleField(null=True)
    longitude = pw.DoubleField(null=True)
    geometry = WktField()

    class Meta:
        database = db
        # unique index on the doublet
        indexes = ((("latitude", "longitude"), True),)


class Parcel(pw.Model):
    numb = pw.TextField()
    county = pw.ForeignKeyField(County, backref="parcels")
    owner = pw.TextField()
    address = pw.TextField()
    data = JSONField()

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

    @property
    def locations(self):
        return [
            permit_permitted_location.permitted_location
            for permit_permitted_location in self.permit_permitted_locations
        ]

    @property
    def regulatory_programs(self):
        return [p.regulatory_program for p in self.permit_regulatory_programs]


class FacilityPermittedLocation(pw.Model):
    facility = pw.ForeignKeyField(Facility, backref="facility_permitted_locations")
    permitted_location = pw.ForeignKeyField(
        PermittedLocation, backref="facility_permitted_locations"
    )
    distance = pw.DoubleField()

    class Meta:
        database = db


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


class PermitRegulatoryProgram(pw.Model):
    permit = pw.ForeignKeyField(Permit, backref="permit_regulatory_programs")
    regulatory_program = pw.ForeignKeyField(
        RegulatoryProgram, backref="permit_regulatory_programs"
    )

    class Meta:
        database = db


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
    def get_images_for_area(cls, lon_min, lat_min, lon_max, lat_max):
        geom = shp.Polygon.from_bounds(lon_min, lat_min, lon_max, lat_max)
        # use postgis index to get images that intersect the area
        images = cls.raw(
            "select * from image where ST_INTERSECTS(geometry, '{}')".format(
                geom.wkt,
            ),
        )
        return list(images)

    @property
    def facilities(self):
        return list(
            {building.facility for building in self.buildings if building.facility}
        )

    @property
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


class AnimalType(pw.Model):
    name = pw.TextField(unique=True)

    class Meta:
        database = db


class FacilityAnimalType(pw.Model):
    facility = pw.ForeignKeyField(Facility, backref="facility_animal_types")
    animal_type = pw.ForeignKeyField(AnimalType, backref="facility_animal_types")
    label_source = pw.TextField(choices=[("human", "human"), ("permit", "permit")])

    class Meta:
        database = db
        indexes = ((("facility", "animal_type"), True),)


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

    class Meta:
        database = db


class RawCloudFactoryImageAnnotation(pw.Model):
    image = pw.ForeignKeyField(
        Image, backref="raw_cloud_factory_image_annotations", unique=True
    )
    created_at = pw.DateTimeField()
    json = JSONField()

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


def make_spatial_indexes():
    for model in models():
        if hasattr(model, "geometry"):
            model.raw(
                f"create index {model._meta.table_name}_geometry_idx on {model._meta.table_name} using gist(geometry)"
            ).execute()


if __name__ == "__main__":
    pass
