import shapely.wkt as wkt
import sqlalchemy as sa
from geoalchemy2 import Geometry
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship

Base = declarative_base()

# Assuming engine creation will be done somewhere else in the code:
# engine = create_engine('postgresql://user:password@localhost/dbname')
# Session = sessionmaker(bind=engine)
# session = Session()


class CountyGroup(Base):
    __tablename__ = "county_group"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

    counties = relationship("County", back_populates="county_group")


class County(Base):
    __tablename__ = "county"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    latitude: Mapped[float]
    longitude: Mapped[float]
    geometry: Mapped[Geometry] = mapped_column(Geometry("POLYGON"))

    county_group_id: Mapped[int] = mapped_column(sa.ForeignKey("county_group.id"))
    county_group = relationship("CountyGroup", back_populates="counties")

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
