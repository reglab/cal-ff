import hashlib
import json
from datetime import datetime

import geoalchemy2 as ga
import rasterio
import shapely as shp
import shapely.wkt as wkt
import sqlalchemy as sa
from geoalchemy2 import Geography, Geometry
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship

from cacafo.transform import DEFAULT_SRID

Base = declarative_base()


class CountyGroup(Base):
    __tablename__ = "county_group"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

    counties: Mapped[list["County"]] = relationship(
        "County", back_populates="county_group"
    )


class County(Base):
    __tablename__ = "county"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    geometry: Mapped[Geography] = mapped_column(
        Geography("MULTIPOLYGON", srid=DEFAULT_SRID)
    )

    county_group_id: Mapped[int] = mapped_column(sa.ForeignKey("county_group.id"))
    county_group: Mapped[CountyGroup] = relationship(
        "CountyGroup", back_populates="counties"
    )

    parcels: Mapped[list["Parcel"]] = relationship("Parcel", back_populates="county")
    images: Mapped[list["Image"]] = relationship("Image", back_populates="county")
    facilities: Mapped[list["Facility"]] = relationship(
        "Facility", back_populates="county"
    )

    @classmethod
    def geocode(cls, session, lon=None, lat=None):
        if None in (lon, lat):
            raise ValueError("None values given to geocode")
        point = wkt.loads(f"POINT({lon} {lat})")
        counties = session.query(cls).all()
        for county in counties:
            if county.geometry.contains(point):
                return county
        raise ValueError(f"No county found for point {point}")


class Parcel(Base):
    __tablename__ = "parcel"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[str]
    address: Mapped[str]
    number: Mapped[str]
    data: Mapped[dict] = mapped_column(JSON)
    inferred_geometry: Mapped[Geography] = mapped_column(
        Geography("POLYGON", srid=DEFAULT_SRID), nullable=True
    )

    county_id: Mapped[int] = mapped_column(sa.ForeignKey("county.id"))
    county = relationship("County", back_populates="parcels")

    registered_location_permits: Mapped[list["Permit"]] = relationship(
        "Permit",
        back_populates="registered_location_parcel",
        foreign_keys="Permit.registered_location_parcel_id",
    )
    geocoded_address_location_permits: Mapped[list["Permit"]] = relationship(
        "Permit",
        back_populates="geocoded_address_location_parcel",
        foreign_keys="Permit.geocoded_address_location_parcel_id",
    )
    buildings: Mapped[list["Building"]] = relationship(
        "Building", back_populates="parcel"
    )

    @property
    def permits(self):
        return self.registered_location_permits + self.geocoded_address_location_permits

    __table_args__ = (sa.UniqueConstraint("number", "county_id"),)


class Permit(Base):
    __tablename__ = "permit"

    id: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)
    registered_location: Mapped[Geography] = mapped_column(
        Geography("POINT", srid=DEFAULT_SRID), nullable=True
    )
    geocoded_address_location: Mapped[Geography] = mapped_column(
        Geography("POINT", srid=DEFAULT_SRID), nullable=True
    )

    registered_location_parcel_id: Mapped[int] = mapped_column(
        sa.ForeignKey("parcel.id"), nullable=True
    )
    registered_location_parcel = relationship(
        "Parcel",
        back_populates="registered_location_permits",
        foreign_keys=[registered_location_parcel_id],
    )
    geocoded_address_location_parcel_id: Mapped[int] = mapped_column(
        sa.ForeignKey("parcel.id"), nullable=True
    )
    geocoded_address_location_parcel = relationship(
        "Parcel",
        back_populates="geocoded_address_location_permits",
        foreign_keys=[geocoded_address_location_parcel_id],
    )

    facility_id: Mapped[int] = mapped_column(
        sa.ForeignKey("facility.id"),
        nullable=True,
    )
    facility = relationship("Facility", back_populates="best_permits")


class Image(Base):
    __tablename__ = "image"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    geometry: Mapped[Geography] = mapped_column(Geography("POLYGON", srid=DEFAULT_SRID))
    bucket: Mapped[str] = mapped_column(sa.String, nullable=True)

    county_id: Mapped[int] = mapped_column(sa.ForeignKey("county.id"))
    county = relationship("County", back_populates="images")

    annotations: Mapped[list["ImageAnnotation"]] = relationship(
        "ImageAnnotation", back_populates="image"
    )

    @property
    def label_status(self):
        if self.bucket is None:
            return "removed"
        if not self.annotations:
            return "unlabeled"
        if self.bucket not in ["0", "1"]:
            return "initially labeled"
        return "labeled"

    @property
    def stratum(self):
        return (self.model_score, self.county.county_group.name)

    @property
    def is_positive(self):
        buildings = sum(
            [list(ann.buildings) for ann in self.annotations if ann.buildings], []
        )
        if not buildings:
            return False
        facilities = {building.facility.id: building.facility for building in buildings}
        for f in facilities.values():
            if f.is_cafo:
                return True
        return False

    @classmethod
    def get_images_for_area(cls, geometry: shp.geometry.base.BaseGeometry, session):
        query = sa.select(cls).where(
            cls.geometry.intersects(geometry) & cls.label_status != "removed"
        )
        return session.execute(query).scalars().all()

    TILE_SIZE = (1024, 1024)

    def from_xy_to_lat_lon(self, geometry, epsg=4326):
        image_geometry = ga.shape.to_shape(self.geometry)
        transform = rasterio.transform.from_bounds(
            *image_geometry.bounds,
            width=self.TILE_SIZE[0],
            height=self.TILE_SIZE[1],
        )
        transformed_geometry = shp.ops.transform(
            lambda x, y: rasterio.transform.xy(transform, y, x),
            geometry,
        )

        if epsg != 4326:
            raise NotImplementedError("Only EPSG 4326 is supported")
        return transformed_geometry

    def get_adjacents(self, session=None, lazy=True, options=None):
        if not session:
            session = sa.orm.object_session(self)
        if lazy:
            query = sa.select(Image).where(
                sa.cast(Image.geometry, ga.Geometry).ST_Touches(
                    sa.cast(self.geometry, ga.Geometry)
                )
                & (Image.bucket.is_not(None))
            )
            if options:
                query = query.options(*options)
            return session.execute(query).unique().scalars().all()
        if not hasattr(Image, "_IMAGES") or (
            options
            and (
                sorted([str(o) for o in getattr(Image, "_OPTIONS", [])])
                != sorted([str(o) for o in options])
            )
        ):
            query = sa.select(Image).where(Image.bucket.is_not(None))
            if options:
                query = query.options(*options)
                Image._OPTIONS = options
            Image._IMAGES = session.execute(query).scalars().unique().all()
        if not hasattr(Image, "_TREE"):
            Image._TREE = shp.strtree.STRtree(
                [ga.shape.to_shape(image.geometry) for image in Image._IMAGES]
            )

        indexes = Image._TREE.query(
            ga.shape.to_shape(self.geometry), predicate="touches"
        )
        return [Image._IMAGES[i] for i in indexes]


class ImageAnnotation(Base):
    __tablename__ = "image_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    annotated_at: Mapped[datetime] = mapped_column(sa.DateTime)
    data: Mapped[dict] = mapped_column(JSON)
    hash: Mapped[str] = mapped_column(
        sa.String,
        default=lambda context: ImageAnnotation._generate_hash_on_insert(context),
        unique=True,
    )

    image_id: Mapped[int] = mapped_column(sa.ForeignKey("image.id"), nullable=True)
    image = relationship("Image", back_populates="annotations")

    buildings: Mapped[list["Building"]] = relationship(
        "Building", back_populates="image_annotation"
    )

    @staticmethod
    def _generate_hash_on_insert(context):
        params = context.get_current_parameters()
        if params["hash"] is None:
            params["hash"] = hashlib.md5(
                json.dumps(params["data"], sort_keys=True).encode()
            ).hexdigest()
        return params["hash"]


class CafoAnnotation(Base):
    __tablename__ = "cafo_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    is_cafo: Mapped[bool]
    is_afo: Mapped[bool]
    location: Mapped[Geography] = mapped_column(Geography("POINT", srid=DEFAULT_SRID))
    annotated_on: Mapped[datetime] = mapped_column(sa.DateTime)
    annotated_by: Mapped[str] = mapped_column(sa.String, nullable=True)

    facility_id: Mapped[int] = mapped_column(
        sa.ForeignKey("facility.id"), nullable=True
    )
    facility = relationship("Facility", back_populates="all_cafo_annotations")


class AnimalTypeAnnotation(Base):
    __tablename__ = "animal_type_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    animal_type: Mapped[str]
    location: Mapped[Geography] = mapped_column(Geography("POINT", srid=DEFAULT_SRID))
    annotated_on: Mapped[datetime] = mapped_column(sa.DateTime)
    annotated_by: Mapped[str] = mapped_column(sa.String)
    notes: Mapped[str] = mapped_column(sa.String)

    facility_id: Mapped[int] = mapped_column(
        sa.ForeignKey("facility.id"), nullable=True
    )
    facility = relationship("Facility", back_populates="all_animal_type_annotations")


class ConstructionAnnotation(Base):
    __tablename__ = "construction_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    location: Mapped[Geography] = mapped_column(Geography("POINT", srid=DEFAULT_SRID))
    construction_lower_bound: Mapped[datetime] = mapped_column(
        sa.DateTime, nullable=True
    )
    construction_upper_bound: Mapped[datetime] = mapped_column(sa.DateTime)
    destruction_lower_bound: Mapped[datetime] = mapped_column(
        sa.DateTime, nullable=True
    )
    destruction_upper_bound: Mapped[datetime] = mapped_column(
        sa.DateTime, nullable=True
    )
    significant_population_change: Mapped[bool] = mapped_column(sa.Boolean)
    is_primarily_indoors: Mapped[bool] = mapped_column(sa.Boolean)
    has_lagoon: Mapped[bool] = mapped_column(sa.Boolean)
    annotated_on: Mapped[datetime] = mapped_column(sa.DateTime)
    data: Mapped[dict] = mapped_column(JSON)

    facility_id: Mapped[int] = mapped_column(
        sa.ForeignKey("facility.id"), nullable=True
    )
    facility = relationship("Facility", back_populates="all_construction_annotations")


class ParcelOwnerNameAnnotation(Base):
    __tablename__ = "parcel_owner_name_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_name: Mapped[str]
    related_owner_name: Mapped[str]
    matched: Mapped[bool]
    annotated_on: Mapped[datetime] = mapped_column(sa.DateTime)
    annotated_by: Mapped[str] = mapped_column(sa.String)


class Building(Base):
    __tablename__ = "building"

    id: Mapped[int] = mapped_column(primary_key=True)
    geometry: Mapped[Geography] = mapped_column(Geography("POLYGON", srid=DEFAULT_SRID))
    image_xy_geometry: Mapped[Geometry] = mapped_column(Geometry("POLYGON"))
    hash: Mapped[str] = mapped_column(
        sa.String,
        default=lambda context: Building._generate_hash_on_insert(context),
        unique=True,
    )

    parcel_id: Mapped[int] = mapped_column(sa.ForeignKey("parcel.id"), nullable=True)
    parcel = relationship("Parcel", back_populates="buildings")

    excluded_at: Mapped[datetime] = mapped_column(sa.DateTime, nullable=True)
    exclude_reason: Mapped[str] = mapped_column(sa.String, nullable=True)

    building_relationships: Mapped[list["BuildingRelationship"]] = relationship(
        "BuildingRelationship",
        foreign_keys="BuildingRelationship.building_id",
        back_populates="building",
    )

    facility_id: Mapped[int] = mapped_column(
        sa.ForeignKey("facility.id"), nullable=True
    )
    facility = relationship("Facility", back_populates="all_buildings")

    image_annotation_id: Mapped[int] = mapped_column(
        sa.ForeignKey("image_annotation.id"), nullable=False
    )
    image_annotation = relationship("ImageAnnotation")

    @staticmethod
    def _generate_hash_on_insert(context):
        params = context.get_current_parameters()
        if params["hash"] is None:
            params["hash"] = hashlib.md5(params["geometry"].encode()).hexdigest()
        return params["hash"]


class BuildingRelationship(Base):
    __tablename__ = "building_relationship"

    id: Mapped[int] = mapped_column(primary_key=True)
    reason: Mapped[str] = mapped_column(sa.String)
    weight: Mapped[float] = mapped_column(sa.Float, nullable=True)

    building_id: Mapped[int] = mapped_column(sa.ForeignKey("building.id"))
    building = relationship("Building", foreign_keys=[building_id])

    related_building_id: Mapped[int] = mapped_column(sa.ForeignKey("building.id"))
    related_building = relationship("Building", foreign_keys=[related_building_id])

    __table_args__ = (sa.CheckConstraint("building_id != related_building_id"),)

    def __str__(self):
        return f"{self.building_id} -> {self.related_building_id} ({self.reason}: {self.weight}))"


class Facility(Base):
    __tablename__ = "facility"

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(
        sa.String,
        default=lambda context: Facility._generate_hash_on_insert(context),
    )
    geometry: Mapped[Geography] = mapped_column(
        Geography("MULTIPOLYGON", srid=DEFAULT_SRID)
    )

    county_id: Mapped[int] = mapped_column(sa.ForeignKey("county.id"), nullable=True)
    county = relationship("County", back_populates="facilities")

    __table_args__ = (
        sa.Index(
            "uq_facility_hash",
            "hash",
            unique=True,
            postgresql_where=sa.text("archived_at IS NULL"),
        ),
    )

    archived_at: Mapped[datetime] = mapped_column(sa.DateTime, nullable=True)

    all_animal_type_annotations: Mapped[list["AnimalTypeAnnotation"]] = relationship(
        "AnimalTypeAnnotation",
        back_populates="facility",
        lazy="selectin",
    )
    all_cafo_annotations: Mapped[list["CafoAnnotation"]] = relationship(
        "CafoAnnotation",
        back_populates="facility",
        lazy="selectin",
    )
    all_construction_annotations: Mapped[list["ConstructionAnnotation"]] = relationship(
        "ConstructionAnnotation",
        back_populates="facility",
        lazy="selectin",
    )
    all_buildings: Mapped[list["Building"]] = relationship(
        "Building",
        back_populates="facility",
        lazy="selectin",
    )
    best_permits: Mapped[list["Permit"]] = relationship(
        "Permit",
        primaryjoin="Permit.facility_id == Facility.id",
    )

    def to_geojson_feature(self):
        geom = ga.shape.to_shape(self.geometry)
        json_geom = shp.geometry.mapping(geom)
        feature = {
            "type": "Feature",
            "geometry": json_geom,
            "id": str(self.hash),
        }

        def d2i(dt):
            return dt and dt.isoformat()

        feature["properties"] = {
            "id": self.id,
            "hash": self.hash,
            "latitude": geom.centroid.y,
            "longitude": geom.centroid.x,
            "lat_min": geom.bounds[1],
            "lon_min": geom.bounds[0],
            "lat_max": geom.bounds[3],
            "lon_max": geom.bounds[2],
            "parcels": [
                {
                    "id": b.parcel.id,
                    "owner": b.parcel.owner,
                    "address": b.parcel.address,
                    "number": b.parcel.number,
                    "county": b.parcel.county.name,
                }
                for b in self.buildings
                if b.parcel
            ],
            # "best_permits": [permit.data for permit in self.permits],
            # "all_permits": [
            #     permit.data
            #     for building in self.buildings
            #     for parcel in [building.parcel]
            #     if parcel
            #     for permit in parcel.permits
            # ],
            "construction_annotation": (
                {
                    "id": self.construction_annotation.id,
                    "construction_lower_bound": d2i(
                        self.construction_annotation.construction_lower_bound
                    ),
                    "construction_upper_bound": d2i(
                        self.construction_annotation.construction_upper_bound
                    ),
                    "destruction_lower_bound": d2i(
                        self.construction_annotation.destruction_lower_bound
                    ),
                    "destruction_upper_bound": d2i(
                        self.construction_annotation.destruction_upper_bound
                    ),
                    "significant_population_change": self.construction_annotation.significant_population_change,
                    "is_primarily_indoors": self.construction_annotation.is_primarily_indoors,
                    "has_lagoon": self.construction_annotation.has_lagoon,
                }
                if self.construction_annotation
                else None
            ),
        }
        feature["bbox"] = list(geom.bounds)
        return feature

    @staticmethod
    def _generate_hash_on_insert(context):
        params = context.get_current_parameters()
        if params["hash"] is None:
            params["hash"] = hashlib.md5(params["geometry"].encode()).hexdigest()
        return params["hash"]

    def all_permits(self, session=None):
        if not session:
            session = sa.orm.object_session(self)
        if not session:
            raise ValueError("Session must be provided for unbound object")
        query = sa.select(Permit).where(
            sa.or_(
                Permit.registered_location_parcel.has(
                    Parcel.geometry.ST_DWithin(self.geometry, 1000)
                ),
                Permit.geocoded_address_location_parcel.has(
                    Parcel.geometry.ST_DWithin(self.geometry, 1000)
                ),
            )
        )
        return session.execute(query).scalars().all()

    @property
    def animal_types(self):
        annotated_types = set(
            [
                annotation.animal_type
                for annotation in self.all_animal_type_annotations
                if annotation.annotated_on
                == max(
                    [
                        annotation.annotated_on
                        for annotation in self.all_animal_type_annotations
                    ]
                )
            ]
        )
        if annotated_types:
            return annotated_types
        all_cow_permits = all(
            "COW" in permit.data.get("Program", "") for permit in self.best_permits
        )
        if all_cow_permits:
            return {"cattle"}
        return set()

    @property
    def is_cafo(self):
        return not any(
            [not annotation.is_cafo for annotation in self.all_cafo_annotations]
        )

    @property
    def is_afo(self):
        return not any(
            [not annotation.is_afo for annotation in self.all_cafo_annotations]
        )

    @property
    def construction_annotation(self):
        return max(
            self.all_construction_annotations,
            key=lambda x: x.annotated_on,
            default=None,
        )

    @property
    def buildings(self):
        return [building for building in self.all_buildings if not building.excluded_at]

    @property
    def images(self):
        return {
            building.image_annotation.image
            for building in self.buildings
            if building.image_annotation and building.image_annotation.image
        }

    @property
    def parcels(self):
        return [building.parcel for building in self.buildings if building.parcel]


class UrbanMask(Base):
    __tablename__ = "urban_mask"

    id: Mapped[int] = mapped_column(primary_key=True)
    uace: Mapped[int] = mapped_column(sa.Integer)
    geoid: Mapped[str] = mapped_column(sa.String)
    name: Mapped[str] = mapped_column(sa.String)
    namelsad: Mapped[str] = mapped_column(sa.String)
    lsad: Mapped[str] = mapped_column(sa.String)
    mtfcc: Mapped[str] = mapped_column(sa.String)
    uatyp: Mapped[str] = mapped_column(sa.String)
    funcstat: Mapped[str] = mapped_column(sa.String)
    aland: Mapped[int] = mapped_column(sa.BigInteger)
    awater: Mapped[int] = mapped_column(sa.BigInteger)
    intptlat: Mapped[str] = mapped_column(sa.String)
    intptlon: Mapped[str] = mapped_column(sa.String)
    geometry: Mapped[Geometry] = mapped_column(Geometry("GEOMETRY", srid=DEFAULT_SRID))


MODELS = Base.__subclasses__()
MODEL_MAP = {model.__tablename__: model for model in MODELS}


def get_model_by_table_name(name):
    return MODEL_MAP[name]
