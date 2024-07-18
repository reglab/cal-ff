from functools import lru_cache

import networkx as nx
import numpy as np
import shapely as shp

from cacafo.db.models import Building


@lru_cache
def mostly_overlapping_buildings():
    buildings = Building.select().order_by(Building.id)
    building_geoms = [b.geometry for b in buildings]
    tree = shp.STRtree(building_geoms)
    overlaps = tree.query(building_geoms, predicate="intersects")
    overlaps = np.vstack(
        (
            overlaps[0][overlaps[0] != overlaps[1]],
            overlaps[1][overlaps[0] != overlaps[1]],
        )
    ).T
    redundant = []
    for idx, intersecting_idx in overlaps:
        building_geom = shp.union_all(shp.ops.polygonize(buildings[idx].geometry))
        intersecting_building_geom = shp.union_all(
            shp.ops.polygonize(buildings[intersecting_idx].geometry)
        )
        intersection = building_geom.intersection(intersecting_building_geom)
        if intersection.area > 0.9 * building_geom.area:
            building = buildings[idx]
            intersecting_building = buildings[intersecting_idx]
            if building_geom.area > intersecting_building_geom.area:
                redundant.append(intersecting_building.id)
            else:
                redundant.append(building.id)
    return set(redundant)


def clean_facility_geometry(facility):
    non_overlapping_buildings = [
        b for b in facility.buildings if b.id not in mostly_overlapping_buildings()
    ]
    geometries = [shp.make_valid(b.geometry) for b in non_overlapping_buildings]
    polygons = []
    while geometries:
        geom = geometries.pop()
        if isinstance(geom, shp.Polygon):
            if geom.area > 0:
                polygons.append(shp.ops.orient(geom))
        elif isinstance(geom, shp.MultiPolygon):
            # only add non overlapping
            geometries.extend(geom.geoms)
        elif isinstance(geom, shp.LineString) or isinstance(geom, shp.MultiLineString):
            try:
                polygons.append(geom.envelope)
            except ValueError as e:
                if "linearring requires at least 4 coordinates" in str(e):
                    continue
                else:
                    raise e
        elif isinstance(geom, shp.GeometryCollection):
            geometries.extend(geom.geoms)
        else:
            raise ValueError(f"Unexpected geometry type {type(geom)}")
    # check for 90% overlaps
    tree = shp.STRtree(polygons)
    overlaps = tree.query(polygons, predicate="intersects")
    overlaps = overlaps.T[overlaps.T[:, 0] < overlaps.T[:, 1]]
    all_overlaps = set(overlaps.flatten())
    final_polygons = set(
        [polygon for idx, polygon in enumerate(polygons) if idx not in all_overlaps]
    )

    components = nx.connected_components(nx.from_edgelist(overlaps))
    for component in components:
        component_polygons = [polygons[idx] for idx in component]
        component_polygon = shp.union_all(component_polygons)
        if isinstance(component_polygon, shp.MultiPolygon):
            final_polygons = final_polygons | set(component_polygon.geoms)
        else:
            final_polygons.add(component_polygon)

    multipolygon = shp.MultiPolygon(list(final_polygons))
    return shp.ops.orient(multipolygon)
