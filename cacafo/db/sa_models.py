import hashlib
from datetime import datetime

import shapely as shp
import shapely.wkt as wkt
import sqlalchemy as sa
from geoalchemy2 import Geometry
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
    geometry: Mapped[Geometry] = mapped_column(Geometry("POLYGON"))

    county_group_id: Mapped[int] = mapped_column(sa.ForeignKey("county_group.id"))
    county_group: Mapped[CountyGroup] = relationship(
        "CountyGroup", back_populates="counties"
    )

    parcels: Mapped[list["Parcel"]] = relationship("Parcel", back_populates="county")

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
    data: Mapped[dict] = mapped_column(sa.JSON)

    county_id: Mapped[int] = mapped_column(sa.ForeignKey("county.id"))
    county = relationship("County", back_populates="parcels")


class Permit(Base):
    __tablename__ = "permit"

    id: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[dict] = mapped_column(sa.JSON)
    registered_location: Mapped[Geometry] = mapped_column(
        Geometry("POINT"), nullable=True
    )
    geocoded_address_location: Mapped[Geometry] = mapped_column(
        Geometry("POINT"), nullable=True
    )

    parcel_id: Mapped[int] = mapped_column(sa.ForeignKey("parcel.id"))
    parcel = relationship("Parcel", back_populates="permits")


class Image(Base):
    __tablename__ = "image"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    geometry: Mapped[Geometry] = mapped_column(Geometry("POLYGON"))
    model_score: Mapped[str]
    label_status: Mapped[str]
    stratum: Mapped[str]

    county_id: Mapped[int] = mapped_column(sa.ForeignKey("county.id"))
    county = relationship("County", back_populates="images")

    @classmethod
    def get_images_for_area(cls, geometry: shp.geometry.base.BaseGeometry, session):
        query = sa.select(cls).where(cls.geometry.intersects(geometry))
        return session.execute(query).scalars().all()


class AnimalTypeAnnotation(Base):
    __tablename__ = "animal_type_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    animal_type: Mapped[str]
    location: Mapped[Geometry] = mapped_column(Geometry("POINT"))


class CafoAnnotation(Base):
    __tablename__ = "cafo_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    is_cafo: Mapped[bool]
    is_afo: Mapped[bool]
    location: Mapped[Geometry] = mapped_column(Geometry("POINT"))


class ConstructionAnnotation(Base):
    __tablename__ = "construction_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    location: Mapped[Geometry] = mapped_column(Geometry("POINT"))
    construction_lower_bound: Mapped[datetime] = mapped_column(sa.DateTime)
    construction_upper_bound: Mapped[datetime] = mapped_column(sa.DateTime)
    destruction_lower_bound: Mapped[datetime] = mapped_column(sa.DateTime)
    destruction_upper_bound: Mapped[datetime] = mapped_column(sa.DateTime)
    significant_population_change: Mapped[bool] = mapped_column(sa.Boolean)
    is_primarily_indoors: Mapped[bool] = mapped_column(sa.Boolean)
    has_lagoon: Mapped[bool] = mapped_column(sa.Boolean)


class ParcelOwnerRelationshipAnnotations(Base):
    __tablename__ = "parcel_owner_name_annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_name: Mapped[str]
    related_owner_name: Mapped[str]


class ImageAnnotation(Base):
    __tablename__ = "image_annotation"

    id: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[dict] = mapped_column(sa.JSON)

    image_id: Mapped[int] = mapped_column(sa.ForeignKey("image.id"))
    image = relationship("Image", back_populates="annotations")


class Building(Base):
    __tablename__ = "building"

    id: Mapped[int] = mapped_column(primary_key=True)
    geometry: Mapped[Geometry] = mapped_column(Geometry("POLYGON"))
    image_xy_geometry: Mapped[Geometry] = mapped_column(Geometry("POLYGON"))

    parcel_id: Mapped[int] = mapped_column(sa.ForeignKey("parcel.id"))
    parcel = relationship("Parcel", back_populates="buildings")

    image_id: Mapped[int] = mapped_column(sa.ForeignKey("image.id"))
    image = relationship("Image", back_populates="buildings")

    excluded_at: Mapped[datetime] = mapped_column(sa.DateTime, nullable=True)
    exclude_reason: Mapped[str] = mapped_column(sa.String, nullable=True)

    building_relationships: Mapped[list["BuildingRelationship"]] = relationship(
        "BuildingRelationship",
        primaryjoin="Building.id == BuildingRelationship.building_id",
        backref="building",
    )
    related_buildings: Mapped[list["Building"]] = relationship(
        "Building",
        secondary=lambda: BuildingRelationship.__table__,
        primaryjoin="Building.id == BuildingRelationship.building_id",
        secondaryjoin="Building.id == BuildingRelationship.related_building_id",
    )


class BuildingRelationship(Base):
    __tablename__ = "building_relationship"

    id: Mapped[int] = mapped_column(primary_key=True)
    reason: Mapped[str] = mapped_column(sa.String)
    weight: Mapped[float] = mapped_column(sa.Float)

    building_id: Mapped[int] = mapped_column(sa.ForeignKey("building.id"))
    building = relationship("Building", foreign_keys=[building_id])

    related_building_id: Mapped[int] = mapped_column(sa.ForeignKey("building.id"))
    related_building = relationship("Building", foreign_keys=[related_building_id])

    __table_args__ = (sa.CheckConstraint("building_id != related_building_id"),)


class Facility(Base):
    __tablename__ = "facility"

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(
        sa.String,
        unique=True,
        default=lambda context: Facility._generate_hash_on_insert(context),
    )
    geometry: Mapped[Geometry] = mapped_column(Geometry("MULTIPOLYGON"))

    county_id: Mapped[int] = mapped_column(sa.ForeignKey("county.id"))
    county = relationship("County", back_populates="facilities")

    @staticmethod
    def _generate_hash_on_insert(context):
        params = context.get_current_parameters()
        if params["hash"] is None:
            params["hash"] = hashlib.md5(params["geometry"].wkt.encode()).hexdigest()
