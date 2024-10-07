import pyproj
import shapely.ops

DEFAULT_SRID = 4326
CA_SRID = 3311

TO_METERS = pyproj.Transformer.from_crs(DEFAULT_SRID, CA_SRID, always_xy=True)
TO_WGS = pyproj.Transformer.from_crs(CA_SRID, DEFAULT_SRID, always_xy=True)


def to_meters(geom):
    return shapely.ops.transform(TO_METERS.transform, geom)


def to_wgs(geom):
    return shapely.ops.transform(TO_WGS.transform, geom)
