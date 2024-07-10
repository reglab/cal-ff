import csv
import functools
import io
import itertools
import json
import os

import diskcache as dc
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import peewee as pw
import scipy as sp
import seaborn as sns
import shapely as shp
import thefuzz
from PIL import Image
from reglab_utils.geo.visualization import add_polygon_layer, create_map
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

import naip
from models import *

FIG_PATH = "figures/"

sns.set(style="whitegrid")
sns.set_palette("Set2")
sns.set_theme("paper", style="white", font="Times New Roman")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (3, 2.5)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}\usepackage{amsmath}"

cache = dc.Cache(".reports_cache")


@functools.cache
def facilities_with_eps(eps):
    return [frozenset(f) for f in BuildingRelationship.facilities(distance=eps)]


@functools.cache
def facilities_with_name_weight(w):
    return [frozenset(f) for f in BuildingRelationship.facilities(tfidf=w)]


@functools.cache
def facilities_set(**kwargs):
    return [frozenset(f) for f in BuildingRelationship.facilities(**kwargs)]


@functools.cache
def facility_group_facility_map_with_eps(eps):
    return {
        facility_group: {
            facility
            for facility in facilities_with_eps(eps)
            if facility.issubset(facility_group)
        }
        for facility_group in facility_groups
    }


def mismatch_to_eps_relationship():
    df = pd.DataFrame()
    df["eps"] = np.arange(0, 1000, 50)
    df["mismatch"] = [
        len(
            [
                facility_group
                for facility_group in facility_groups
                if facility_group_facility_map_with_eps(e)[facility_group]
                != facility_group_old_facility_map[facility_group]
            ]
        )
        for e in tqdm(df["eps"])
    ]
    # add dots
    sns.lineplot(x="eps", y="mismatch", data=df)
    sns.scatterplot(x="eps", y="mismatch", data=df)
    plt.title("Mismatching Facility Groups vs. DBSCAN epsilon")
    plt.xlabel("DBSCAN epsilon")
    plt.ylabel("Mismatching Facility Groups")
    plt.savefig(os.path.join(FIG_PATH, "mismatch_to_eps.png"))


def num_facilities_to_eps_relationship(tfidf=700, fuzzy=600):
    df = pd.DataFrame()
    df["eps"] = np.arange(0, 1000, 20)
    df["num_facilities"] = [
        len(facilities_set(distance=e, tfidf=tfidf, fuzzy=fuzzy))
        for e in tqdm(df["eps"])
    ]
    # add dots
    sns.lineplot(x="eps", y="num_facilities", data=df)
    sns.scatterplot(x="eps", y="num_facilities", data=df)
    plt.title("Number of Facility Groups vs. DBSCAN max distance")
    plt.xlabel("Max Distance (m)")
    plt.ylabel("Number of Facility Groups")
    plt.savefig(os.path.join(FIG_PATH, f"num_facilities_to_eps_t{tfidf}_f{fuzzy}.png"))
    plt.clf()


def tf_idf_examples(num_per=100):
    ranges = zip(range(0, 1000, 50), range(50, 1000, 50))
    rows = []
    for low, high in ranges:
        OtherBuilding = Building.alias()
        OtherParcel = Parcel.alias()
        examples = (
            BuildingRelationship.select(
                BuildingRelationship.weight.alias("weight"),
                Parcel.owner.alias("owner_1"),
                OtherParcel.owner.alias("owner_2"),
            )
            .join(Building, on=(Building.id == BuildingRelationship.building))
            .join(
                OtherBuilding,
                on=(OtherBuilding.id == BuildingRelationship.other_building),
            )
            .join(Parcel, on=(Parcel.id == Building.parcel))
            .join(OtherParcel, on=(OtherParcel.id == OtherBuilding.parcel))
            .where(
                (BuildingRelationship.reason == "parcel name tf-idf")
                & (BuildingRelationship.weight >= low)
                & (BuildingRelationship.weight < high)
            )
            .distinct()
            .limit(num_per)
            .dicts()
        )
        rows.extend(examples)
    rows = sorted(rows, key=lambda x: x["weight"])
    writer = csv.DictWriter(
        open("outputs/parcel_name_tfidf_examples.csv", "w"),
        fieldnames=["weight", "owner_1", "owner_2"],
    )
    writer.writeheader()
    writer.writerows(rows)


def fuzzy_examples(num_per=100):
    ranges = zip(range(0, 1000, 50), range(50, 1000, 50))
    rows = []
    for low, high in ranges:
        OtherBuilding = Building.alias()
        OtherParcel = Parcel.alias()
        examples = (
            BuildingRelationship.select(
                BuildingRelationship.weight.alias("weight"),
                Parcel.owner.alias("owner_1"),
                OtherParcel.owner.alias("owner_2"),
            )
            .join(Building, on=(Building.id == BuildingRelationship.building))
            .join(
                OtherBuilding,
                on=(OtherBuilding.id == BuildingRelationship.other_building),
            )
            .join(Parcel, on=(Parcel.id == Building.parcel))
            .join(OtherParcel, on=(OtherParcel.id == OtherBuilding.parcel))
            .where(
                (BuildingRelationship.reason == "parcel name fuzzy")
                & (BuildingRelationship.weight >= low)
                & (BuildingRelationship.weight < high)
            )
            .limit(num_per)
            .distinct()
            .dicts()
        )
        rows.extend(examples)
    rows = sorted(rows, key=lambda x: x["weight"])
    writer = csv.DictWriter(
        open("outputs/parcel_name_fuzzy_examples.csv", "w"),
        fieldnames=["weight", "owner_1", "owner_2"],
    )
    writer.writeheader()
    writer.writerows(rows)


def parcels_per_facility_histogram(**kwargs):
    parcels = {
        row["id"]: row["parcel"]
        for row in Building.select(Building.id, Building.parcel).dicts()
    }
    parcels = [set([parcels[b] for b in f]) for f in facilities_set(**kwargs)]
    num_parcels = [len(p) for p in parcels]
    print("Number of facilities: {}".format(len(num_parcels)))
    for p in parcels:
        if len(p) > 2:
            # get parcel owners
            owners = {Parcel.get(Parcel.id == parcel).owner for parcel in p}
            print("Parcel owners: {}".format(set(owners)))
    df = pd.DataFrame()
    df["num_parcels"] = num_parcels
    # log y
    sns.histplot(data=df, x="num_parcels", bins=50, log_scale=(False, True))
    plt.title("Number of Parcels per Facility")
    plt.xlabel("Number of Parcels")
    plt.ylabel("Number of Facilities")
    plt.savefig(
        os.path.join(
            FIG_PATH,
            "num_parcels_per_facility_d{}_t{}_f{}_wannotations.png".format(
                kwargs.get("distance", 400),
                kwargs.get("tfidf", 700),
                kwargs.get("fuzzy", 600),
            ),
        )
    )
    plt.clf()


def midrange_parcel_name_matches():
    aggressive_group = facilities_set(distance=400, tfidf=-1, fuzzy=350)
    conservative_group = set(facilities_set(distance=400, tfidf=700, fuzzy=600))

    print("Aggressive group size: {}".format(len(aggressive_group)))
    print("Conservative group size: {}".format(len(conservative_group)))

    facility_splits = {}
    for f in aggressive_group:
        to_remove = set()
        for f2 in conservative_group:
            if f.issuperset(f2):
                facility_splits[f] = facility_splits.get(f, set()) | {f2}
                to_remove.add(f2)
        conservative_group -= to_remove

    building_owner = {
        row["id"]: row["owner"]
        for row in Building.select(Building.id, Parcel.owner.alias("owner"))
        .join(Parcel)
        .dicts()
    }
    writer = csv.DictWriter(
        open("outputs/parcel_name_medium_matches.csv", "w"),
        fieldnames=["owner_1", "owner_2"],
    )
    writer.writeheader()
    for super_facility, sub_facilities in facility_splits.items():
        if len(sub_facilities) == 0:
            raise ValueError(
                "No sub facilities found for super facility: {}".format(super_facility)
            )
        if len(sub_facilities) == 1:
            continue
        parcel_owners_in_super_facility = {building_owner[b] for b in super_facility}
        parcel_owners_in_sub_facilities = [
            {building_owner[b] for b in f} for f in sub_facilities
        ]
        print(
            "Super facility parcel owners: {}".format(parcel_owners_in_super_facility)
        )
        possible_matches = itertools.combinations(parcel_owners_in_super_facility, 2)
        for pm in possible_matches:
            weight = BuildingRelationship.fuzzy_name_match(pm[0], pm[1])
            if weight > 350:
                writer.writerow({"owner_1": pm[0], "owner_2": pm[1]})


def high_tfidf_low_fuzzy():
    high_tfidf_low_fuzzy = facilities_set(
        distance=200, tfidf=900, fuzzy=-1, fuzzy_max=100
    )
    building_owner = {
        row["id"]: row["owner"]
        for row in Building.select(Building.id, Parcel.owner.alias("owner"))
        .join(Parcel)
        .dicts()
    }
    for f in high_tfidf_low_fuzzy:
        parcel_owners = {building_owner[b] for b in f if b in building_owner}
        if len(parcel_owners) == 1:
            continue
        print("Parcel owners: {}".format(parcel_owners))


def low_tfidf_high_fuzzy():
    low_tfidf_high_fuzzy = facilities_set(
        distance=200, tfidf=-1, tfidf_max=100, fuzzy=900
    )
    building_owner = {
        row["id"]: row["owner"]
        for row in Building.select(Building.id, Parcel.owner.alias("owner"))
        .join(Parcel)
        .dicts()
    }
    for f in low_tfidf_high_fuzzy:
        parcel_owners = {building_owner[b] for b in f if b in building_owner}
        if len(parcel_owners) == 1:
            continue
        print("Parcel owners: {}".format(parcel_owners))


def num_facilities_to_name_tfidf():
    df = pd.DataFrame()
    df["weight"] = np.arange(0, 1000, 20)
    df["num_facilities"] = [
        len(facilities_set(tfidf=w, distance=400, fuzzy=-1)) for w in tqdm(df["weight"])
    ]
    # add dots
    sns.lineplot(x="weight", y="num_facilities", data=df)
    sns.scatterplot(x="weight", y="num_facilities", data=df)
    plt.title("Number of Facilities vs. TF-IDF weight")
    plt.xlabel("TF-IDF Weight")
    plt.ylabel("Number of Facilities")
    plt.savefig(os.path.join(FIG_PATH, "num_facilities_to_tfidf_weight.png"))
    plt.clf()


def num_facilities_to_name_fuzzy():
    df = pd.DataFrame()
    df["weight"] = np.arange(0, 1000, 20)
    df["num_facilities"] = [
        len(facilities_set(fuzzy=w, distance=200, tfidf=-1)) for w in tqdm(df["weight"])
    ]
    # add dots
    sns.lineplot(x="weight", y="num_facilities", data=df)
    sns.scatterplot(x="weight", y="num_facilities", data=df)
    plt.title("Number of Facilities vs. Fuzzy match weight")
    plt.xlabel("Fuzz Ratio * 10")
    plt.ylabel("Number of Facility Groups")
    plt.savefig(os.path.join(FIG_PATH, "num_facilities_to_fuzz.png"))
    plt.clf()


def num_facilities_to_eps_comparison():
    df = pd.DataFrame()
    df["distance"] = np.arange(0, 1000, 50)
    df["num_facilities_without_parcel_owner_matching"] = [
        len(facilities_set(distance=e, tfidf=-1, fuzzy=-1))
        for e in tqdm(df["distance"])
    ]
    df["num_facilities_with_parcel_owner_matching"] = [
        len(facilities_set(distance=e)) for e in tqdm(df["distance"])
    ]
    sns.lineplot(
        x="distance", y="num_facilities_without_parcel_owner_matching", data=df
    )
    # sns.scatterplot(x="distance", y="num_facilities_without_parcel_owner_matching", data=df)
    sns.lineplot(x="distance", y="num_facilities_with_parcel_owner_matching", data=df)
    # sns.scatterplot(x="distance", y="num_facilities_with_parcel_owner_matching", data=df)
    plt.title(
        "Number of Facilities vs. distance threshold, before and after parcel owner matching"
    )
    plt.xlabel("DBSCAN distance threshold (m)")
    plt.ylabel("Number of facilities")
    plt.savefig(os.path.join(FIG_PATH, "num_facilities_to_eps_comparison.png"))
    plt.clf()


def num_buildings_per_facility():
    df = pd.DataFrame()
    df["num_buildings"] = [len(f) for f in BuildingRelationship.facilities()]
    sns.histplot(data=df, x="num_buildings", bins=50, log_scale=(False, True))
    plt.title("Number of Buildings per Facility")
    plt.xlabel("Number of Buildings")
    plt.ylabel("Number of Facilities")
    plt.savefig(os.path.join(FIG_PATH, "num_buildings_per_facility.png"))
    plt.clf()


def map_building_set(building_set, map_=None, bbox_color="blue", building_color="red"):
    building_geoms = sum(
        Building.select(Building.geometry)
        .where(Building.id.in_(building_set))
        .tuples(),
        (),
    )
    gdf = gpd.GeoDataFrame()
    gdf["geometry"] = building_geoms
    bbox = gdf.total_bounds
    bbox_gdf = gpd.GeoDataFrame()
    bbox_gdf["geometry"] = [shp.geometry.box(*bbox)]
    lon, lat = gdf.centroid.iloc[0].coords[0]
    if map_ is None:
        map_ = create_map(lat, lon, map_kwargs={"control_scale": True})
    add_polygon_layer(
        map_, gdf, layer_name="buildings", polygon_kwargs={"fill_color": building_color}
    )
    add_polygon_layer(
        map_, bbox_gdf, layer_name="bbox", polygon_kwargs={"fill_color": bbox_color}
    )
    map_.fit_bounds(map_.get_bounds())
    return map_


def map_facility(facility, map_=None, building_color="red", bbox_color="blue"):
    lon, lat = facility.longitude, facility.latitude
    if map_ is None:
        map_ = create_map(lat, lon, map_kwargs={"control_scale": True})
    gdf = gpd.GeoDataFrame()
    gdf["geometry"] = [facility.geometry]
    add_polygon_layer(
        map_, gdf, layer_name="facility", polygon_kwargs={"fill_color": building_color}
    )
    bbox_gdf = gpd.GeoDataFrame()
    bbox_gdf["geometry"] = [facility.geometry.envelope]
    add_polygon_layer(
        map_, bbox_gdf, layer_name="bbox", polygon_kwargs={"fill_color": bbox_color}
    )
    map_.fit_bounds(map_.get_bounds())
    return map_


def map_facilities_with_most_parcels(**kwargs):
    facilities = facilities_set(**kwargs)
    parcels = {
        row["id"]: row["parcel"]
        for row in Building.select(Building.id, Building.parcel).dicts()
    }
    parcel_sets = [set([parcels[b] for b in f]) for f in facilities]
    facilities_by_parcel_length = sorted(
        zip(facilities, parcel_sets), key=lambda x: len(x[1]), reverse=True
    )
    # make directory
    os.makedirs("maps/facility_with_most_parcels", exist_ok=True)
    for i, (f, p) in enumerate(facilities_by_parcel_length[:10]):
        print("Facility with most parcels: {}".format(len(p)))
        parcel_owners = {
            Parcel.get(Parcel.id == parcel).owner for parcel in p if parcel in parcels
        }
        print("Parcel owners: {}".format(parcel_owners))
        map_ = map_building_set(f)
        map_data = map_._to_png(1)
        img = Image.open(io.BytesIO(map_data))
        owner_short = sorted(list(parcel_owners), key=len)[-1].split(" ")[0]
        img.save(
            f"maps/facility_with_most_parcels/{i}_{list(parcel_owners)[0].split(' ')[0]}_{len(p)}.png"
        )


def map_facilities_with_most_buildings(**kwargs):
    facilities = facilities_set(**kwargs)
    facilities_by_building_length = sorted(
        facilities, key=lambda x: len(x), reverse=True
    )
    # make directory
    os.makedirs("maps/facility_with_most_buildings", exist_ok=True)
    for i, f in enumerate(facilities_by_building_length[:5]):
        print("Facility with most buildings: {}".format(len(f)))
        map_ = map_building_set(f)
        map_data = map_._to_png(1)
        img = Image.open(io.BytesIO(map_data))
        img.save(f"maps/facility_with_most_buildings/{i}_{len(f)}.png")


def map_brandt():
    buildings = set(
        sum(
            Building.select(Building.id)
            .join(Parcel)
            .where(Parcel.owner % "%BRANDT%")
            .tuples(),
            (),
        )
    )
    map_ = map_building_set(buildings)
    map_.save("maps/brandt.html")


def facilities_without_animal_types():
    facilities = (
        Facility.select()
        .join(FacilityAnimalType, join_type=pw.JOIN.LEFT_OUTER)
        .group_by(Facility.id)
        .having(pw.fn.Count(FacilityAnimalType.id) == 0)
    )
    # print("Facilities without animal types: {}".format(len(facilities)))
    return {f.id for f in facilities}


def map_facilities_without_animal_types():
    facilities = facilities_without_animal_types()
    for i, f in enumerate(facilities):
        print("Facility without animal types: {}".format(f))
        map_ = map_facility(f)
        map_.save(f"maps/facilities_without_animal_types/{f.id}.html")


def num_buildings_by_animal_type():
    rows = (
        Facility.select(
            Facility.id.alias("facility_id"),
            Facility.geometry.alias("geometry"),
            pw.fn.Count(Building.id).alias("n_buildings"),
            AnimalType.name.alias("animal_type"),
        )
        .join(Building)
        .join(FacilityAnimalType, on=(FacilityAnimalType.facility == Facility.id))
        .join(AnimalType)
        .group_by(Facility.id, AnimalType.id)
        .order_by(Facility.id)
        .dicts()
    )
    df = gpd.GeoDataFrame(rows)
    df.crs = "EPSG:4326"
    df.to_crs("EPSG:3311", inplace=True)
    df["area"] = df["geometry"].envelope.area
    df = df[df["animal_type"].isin(["cows", "dairy"])]
    # mean values within the same 5 building num
    df["groups"] = df["n_buildings"] // 5
    df = df.groupby(["groups", "animal_type"]).mean().reset_index()
    sns.scatterplot(data=df, x="n_buildings", y="area", hue="animal_type")
    plt.yscale("log")
    plt.title("Number of Buildings by Area, binned")
    plt.xlabel("Number of Buildings")
    plt.ylabel("Area (m^2)")
    plt.savefig(os.path.join(FIG_PATH, "n_buildings_by_animal_type.png"))
    plt.clf()


def permits_without_locations():
    permits = (
        Permit.select()
        .join(
            PermitPermittedLocation,
            pw.JOIN.LEFT_OUTER,
        )
        .where(PermitPermittedLocation.id.is_null())
    )
    print("Permits without locations: {}".format(len(permits)))
    return list(permits)


def facilities_without_permits():
    facilities = (
        Facility.select()
        .join(FacilityPermittedLocation, pw.JOIN.LEFT_OUTER)
        .where(FacilityPermittedLocation.id.is_null())
    )
    print("Facilities without permits: {}".format(len(facilities)))
    return list(facilities)


def distance_to_closest_facility():
    facility_geometries = Facility.select(Facility.id, Facility.geometry).dicts()
    gdf = gpd.GeoDataFrame(facility_geometries)
    gdf.crs = "EPSG:4326"
    gdf.to_crs("EPSG:3311", inplace=True)

    index = shp.STRtree(gdf["geometry"].values)
    distances = []
    for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
        idx, dist = index.query_nearest(
            row["geometry"], max_distance=1000, return_distance=True, exclusive=True
        )
        dist = list(dist)
        if dist:
            dist = dist[0]
            distances.append(dist)
    sns.histplot(distances, bins=50)
    plt.title("Distance to closest facility")
    plt.xlabel("Distance (m)")
    plt.ylabel("Number of facilities")
    plt.savefig(os.path.join(FIG_PATH, "distance_to_closest_facility.png"))


def map_closest_building_sets(**kwargs):
    building_geometries = {
        row["id"]: row["geometry"]
        for row in Building.select(Building.id, Building.geometry).dicts()
    }
    building_sets = facilities_set(**kwargs)
    facility_geometries = [
        {"id": i, "geometry": shp.MultiPolygon([building_geometries[b] for b in f])}
        for i, f in enumerate(building_sets)
    ]
    gdf = gpd.GeoDataFrame(facility_geometries)
    gdf.crs = "EPSG:4326"
    gdf.to_crs("EPSG:3311", inplace=True)
    index = shp.STRtree(gdf["geometry"].values)
    facility_distances = []
    for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
        idx, dist = index.query_nearest(
            row["geometry"], max_distance=1000, return_distance=True, exclusive=True
        )
        dist = list(dist)
        if dist:
            dist = dist[0]
            facility_distances.append(
                {
                    "id": row["id"],
                    "other_facility_id": gdf.loc[idx, "id"].values[0],
                    "distance": dist,
                }
            )
    facility_distances = sorted(
        facility_distances, key=lambda x: x["distance"], reverse=True
    )
    plotted = set()
    dist = 0
    i = 0
    # make directory
    os.makedirs("maps/closest_building_sets", exist_ok=True)
    while facility_distances and dist < 100:
        fd = facility_distances.pop()
        dist = fd["distance"]
        print("Closest facility: {}".format(fd))
        key = tuple(sorted((fd["id"], fd["other_facility_id"])))
        if key in plotted:
            continue
        map_ = map_building_set(
            building_sets[fd["id"]],
            building_color="red",
            bbox_color="red",
        )
        map_ = map_building_set(
            building_sets[fd["other_facility_id"]],
            map_=map_,
            building_color="blue",
            bbox_color="blue",
        )
        plotted.add(key)
        map_.get_root().html.add_child(
            folium.Element(
                f"<h1>Facility {fd['id']} to Facility {fd['other_facility_id']} at {int(dist)}m</h1>"
            )
        )
        map_data = map_._to_png(1)
        img = Image.open(io.BytesIO(map_data))
        img.save(f"maps/closest_building_sets/{i}_{int(dist)}.png")
        i += 1


def map_lone_building_sets(**kwargs):
    building_geometries = {
        row["id"]: row["geometry"]
        for row in Building.select(Building.id, Building.geometry).dicts()
    }
    building_sets = facilities_set(**kwargs)
    facility_geometries = [
        {
            "id": i,
            "geometry": shp.MultiPolygon([building_geometries[b] for b in f]),
            "building_ids": f,
        }
        for i, f in enumerate(building_sets)
    ]
    gdf = gpd.GeoDataFrame(facility_geometries)
    gdf.crs = "EPSG:4326"
    gdf.to_crs("EPSG:3311", inplace=True)
    index = shp.STRtree(gdf["geometry"].values)

    onebuilding = gdf[gdf["building_ids"].apply(len) == 1]
    onebuilding.reset_index(inplace=True)
    (_, onebuilding["nearest_idx"]), onebuilding[
        "nearest_distance"
    ] = index.query_nearest(
        onebuilding["geometry"].values,
        # max_distance=1000,
        return_distance=True,
        exclusive=True,
        all_matches=False,
    )
    onebuilding.sort_values("nearest_distance", inplace=True, ascending=True)
    onebuilding.reset_index(inplace=True)

    notes_csv = open("outputs/lone_building_sets.csv", "w")
    note_file = csv.DictWriter(
        notes_csv,
        fieldnames=[
            "building_id",
            "nearest_building_id",
            "distance",
            "merge",
            "delete",
            "no_action",
        ],
    )
    note_file.writeheader()

    # make directory
    os.makedirs("maps/lone_building_sets", exist_ok=True)
    for i, row in tqdm(onebuilding.iterrows(), total=len(onebuilding)):
        map_ = map_building_set(
            row["building_ids"],
            building_color="red",
            bbox_color="red",
        )
        map_ = map_building_set(
            gdf.loc[row["nearest_idx"], "building_ids"],
            map_=map_,
            building_color="blue",
            bbox_color="blue",
        )
        # add building id
        map_.get_root().html.add_child(
            folium.Element(f"<h1>Building {row['building_ids']}</h1>")
        )
        map_.get_root().html.add_child(
            folium.Element(
                f"<h1>Closest facility at distance {int(row['nearest_distance'])}m</h1>"
            )
        )
        # add link to next element
        map_.get_root().html.add_child(folium.Element(f"<a href='{i+1}.html'>Next</a>"))
        map_data = map_._to_png(1)
        img = Image.open(io.BytesIO(map_data))
        img.save(f"maps/lone_building_sets/{i}.png")

        note_file.writerow(
            {
                "building_id": list(row["building_ids"])[0],
                "nearest_building_id": list(
                    gdf.loc[row["nearest_idx"], "building_ids"]
                )[0],
                "distance": int(row["nearest_distance"]),
                "merge": 0,
                "delete": 0,
                "no_action": 0,
            }
        )
    notes_csv.close()


def map_grouped_lone_building_sets():
    building_geometries = {
        row["id"]: row["geometry"]
        for row in Building.select(Building.id, Building.geometry).dicts()
    }
    building_sets = facilities_set(lone_building_distance=0)
    facility_geometries = [
        {
            "id": i,
            "geometry": shp.MultiPolygon([building_geometries[b] for b in f]),
            "building_ids": f,
        }
        for i, f in enumerate(building_sets)
    ]
    gdf = gpd.GeoDataFrame(facility_geometries)
    gdf.crs = "EPSG:4326"
    gdf.to_crs("EPSG:3311", inplace=True)
    index = shp.STRtree(gdf["geometry"].values)

    onebuilding = gdf[gdf["building_ids"].apply(len) == 1]
    onebuilding.reset_index(inplace=True)
    (_, onebuilding["nearest_idx"]), onebuilding[
        "nearest_distance"
    ] = index.query_nearest(
        onebuilding["geometry"].values,
        # max_distance=1000,
        return_distance=True,
        exclusive=True,
        all_matches=False,
    )
    onebuilding.sort_values("nearest_distance", inplace=True, ascending=True)
    onebuilding.reset_index(inplace=True)
    on_building = onebuilding[onebuilding["nearest_distance"] < 50]

    # make directory
    os.makedirs("maps/grouped_lone_building_sets", exist_ok=True)
    for i, row in tqdm(onebuilding.iterrows(), total=len(onebuilding)):
        map_ = map_building_set(
            row["building_ids"],
            building_color="red",
            bbox_color="red",
        )
        map_ = map_building_set(
            gdf.loc[row["nearest_idx"], "building_ids"],
            map_=map_,
            building_color="blue",
            bbox_color="blue",
        )
        # add building id
        map_.get_root().html.add_child(
            folium.Element(f"<h1>Building {row['building_ids']}</h1>")
        )
        map_.get_root().html.add_child(
            folium.Element(
                f"<h1>Closest facility at distance {int(row['nearest_distance'])}m</h1>"
            )
        )
        # add link to next element
        map_.get_root().html.add_child(folium.Element(f"<a href='{i+1}.html'>Next</a>"))
        map_data = map_._to_png(1)
        img = Image.open(io.BytesIO(map_data))
        img.save(f"maps/grouped_lone_building_sets/{i}.png")


def map_closest_facilities():
    facility_geometries = Facility.select(Facility.id, Facility.geometry).dicts()
    gdf = gpd.GeoDataFrame(facility_geometries)
    gdf.crs = "EPSG:4326"
    gdf.to_crs("EPSG:3311", inplace=True)
    index = shp.STRtree(gdf["geometry"].values)
    facility_distances = []
    for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
        idx, dist = index.query_nearest(
            row["geometry"], max_distance=1000, return_distance=True, exclusive=True
        )
        dist = list(dist)
        if dist:
            dist = dist[0]
            facility_distances.append(
                {
                    "id": row["id"],
                    "other_facility_id": gdf.loc[idx, "id"].values[0],
                    "distance": dist,
                }
            )
    facility_distances = sorted(
        facility_distances, key=lambda x: x["distance"], reverse=True
    )
    plotted = set()
    dist = 0
    i = 0
    facility_owners = {}
    # make directory
    os.makedirs("maps/closest_facilities", exist_ok=True)
    for row in (
        Facility.select(Facility.id, Parcel.owner).join(Building).join(Parcel).dicts()
    ):
        facility_owners[row["id"]] = facility_owners.get(row["id"], set()) | {
            row["owner"]
        }
    while facility_distances and dist < 100:
        fd = facility_distances.pop()
        dist = fd["distance"]
        print("Closest facility: {}".format(fd))
        key = tuple(sorted((fd["id"], fd["other_facility_id"])))
        if key in plotted:
            continue
        map_ = map_facility(
            Facility.get(Facility.id == fd["id"]),
            building_color="red",
            bbox_color="red",
        )
        map_ = map_facility(
            Facility.get(Facility.id == fd["other_facility_id"]),
            map_=map_,
            building_color="blue",
            bbox_color="blue",
        )
        plotted.add(key)
        map_.get_root().html.add_child(
            folium.Element(
                f"<h1>Facility {fd['id']} to Facility {fd['other_facility_id']} at {int(dist)}m</h1>"
            )
        )
        owners = facility_owners.get(fd["id"], None)
        map_.get_root().html.add_child(
            folium.Element(f"<h2>Facility {fd['id']} owners: {owners}</h2>")
        )
        owners = facility_owners.get(fd["other_facility_id"], None)
        map_.get_root().html.add_child(
            folium.Element(
                f"<h2>Facility {fd['other_facility_id']} owners: {owners}</h2>"
            )
        )
        map_data = map_._to_png(1)
        img = Image.open(io.BytesIO(map_data))
        img.save(f"maps/closest_facilities/{i}_{int(dist)}.png")
        i += 1


def map_closest_facilities_with_owners():
    facility_geometries = Facility.select(Facility.id, Facility.geometry).dicts()
    gdf = gpd.GeoDataFrame(facility_geometries)
    gdf.crs = "EPSG:4326"
    gdf.to_crs("EPSG:3311", inplace=True)
    index = shp.STRtree(gdf["geometry"].values)
    facility_distances = []
    # make directory
    os.makedirs("maps/closest_facilities_with_owners", exist_ok=True)
    for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
        idx, dist = index.query_nearest(
            row["geometry"], max_distance=1000, return_distance=True, exclusive=True
        )
        dist = list(dist)
        if dist:
            dist = dist[0]
            facility_distances.append(
                {
                    "id": row["id"],
                    "other_facility_id": gdf.loc[idx, "id"].values[0],
                    "distance": dist,
                }
            )
    facility_distances = sorted(
        facility_distances, key=lambda x: x["distance"], reverse=True
    )
    plotted = set()
    dist = 0
    i = 0
    facility_owners = {}
    # make directory
    os.makedirs("maps/closest_facilities", exist_ok=True)
    for row in (
        Facility.select(Facility.id, Parcel.owner).join(Building).join(Parcel).dicts()
    ):
        facility_owners[row["id"]] = facility_owners.get(row["id"], set()) | {
            row["owner"]
        }
    while facility_distances and dist < 400:
        fd = facility_distances.pop()
        dist = fd["distance"]
        print("Closest facility: {}".format(fd))
        key = tuple(sorted((fd["id"], fd["other_facility_id"])))
        if key in plotted:
            continue
        map_ = map_facility(
            Facility.get(Facility.id == fd["id"]),
            building_color="red",
            bbox_color="red",
        )
        map_ = map_facility(
            Facility.get(Facility.id == fd["other_facility_id"]),
            map_=map_,
            building_color="blue",
            bbox_color="blue",
        )
        plotted.add(key)
        map_.get_root().html.add_child(
            folium.Element(
                f"<h1>Facility {fd['id']} to Facility {fd['other_facility_id']} at {int(dist)}m</h1>"
            )
        )
        owners = facility_owners.get(fd["id"], None)
        if not owners:
            continue
        map_.get_root().html.add_child(
            folium.Element(f"<h2>Facility {fd['id']} owners: {owners}</h2>")
        )
        owners = facility_owners.get(fd["other_facility_id"], None)
        if not owners:
            continue
        map_.get_root().html.add_child(
            folium.Element(
                f"<h2>Facility {fd['other_facility_id']} owners: {owners}</h2>"
            )
        )
        map_data = map_._to_png(1)
        img = Image.open(io.BytesIO(map_data))
        img.save(f"maps/closest_facilities_with_owners/{i}_{int(dist)}.png")
        i += 1


def mismatching_permits_per_distance():
    facilities_permits = list(
        Facility.select(
            Facility.id.alias("facility_id"),
            FacilityPermittedLocation.distance.alias("distance"),
            Permit.facility_name.alias("facility_name"),
            Permit.facility_address.alias("facility_address"),
        )
        .join(FacilityPermittedLocation)
        .join(PermittedLocation)
        .join(PermitPermittedLocation)
        .join(Permit)
        .dicts()
    )
    facilities_permits = pd.DataFrame.from_records(facilities_permits)

    def permits_are_mismatching(rows, threshold=900):
        if len(rows) == 1:
            return False
        names = set(rows["facility_name"].values)
        if any([not n for n in names]):
            names = set(rows["facility_address"].values)
            if any([not n for n in names]):
                raise ValueError("No names or addresses for all permits")
        if all(
            [
                BuildingRelationship.fuzzy_name_match(pair1, pair2) > threshold
                for pair1, pair2 in itertools.combinations(names, 2)
            ]
        ):
            return False
        return True

    distances = list(range(50, 1050, 50))
    counts = {t: list() for t in range(100, 1000, 100)}
    total_facilities_with_matched_permits = []
    for d in distances:
        fp = facilities_permits.loc[facilities_permits["distance"] <= d, :]
        total_facilities_with_matched_permits.append(len(fp["facility_id"].unique()))
        for c, l in counts.items():
            fp1 = fp.groupby("facility_id").filter(
                lambda row: permits_are_mismatching(row, threshold=c)
            )
            l.append(len(fp1["facility_id"].unique()))

    df = pd.DataFrame()
    df["distance"] = distances
    df["num_facilities_with_matched_permits"] = total_facilities_with_matched_permits
    df["num_mismatching_permits"] = counts[600]
    sns.lineplot(
        x="distance", y="num_mismatching_permits", data=df, label="# mismatching"
    )
    sns.scatterplot(x="distance", y="num_mismatching_permits", data=df)

    # plot sensitivity for counts
    for c, l in counts.items():
        kwargs = (
            {"label": "# mismatching, other fuzz thresh".format(c)} if c == 900 else {}
        )
        ax = sns.lineplot(x="distance", y=l, data=df, color="grey", alpha=0.5, **kwargs)
        ax.lines[-1].set_linestyle("--")

    sns.lineplot(
        x="distance",
        y="num_facilities_with_matched_permits",
        data=df,
        label="# matched",
    )
    sns.scatterplot(x="distance", y="num_facilities_with_matched_permits", data=df)
    plt.legend(fontsize="small")
    plt.title("Number of facilities with mismatching permits vs. distance")
    plt.xlabel("Distance (m)")
    plt.ylabel("Number of facilities")
    plt.savefig(os.path.join(FIG_PATH, "mismatching_permits_per_distance.png"))
    plt.clf()


def n_bridges_per_facility():
    df = pd.DataFrame()
    graph = BuildingRelationship.graph()
    bridges = list(nx.bridges(graph))
    facilities = list(nx.connected_components(graph))
    df["n_bridges"] = [
        len([b for b in bridges if b[0] in f and b[1] in f]) for f in tqdm(facilities)
    ]
    sns.histplot(data=df, x="n_bridges", bins=5, log_scale=(False, True))
    plt.title("Number of Bridges per Facility")
    plt.xlabel("Number of Bridges")
    plt.ylabel("Number of Facilities")
    plt.savefig(os.path.join(FIG_PATH, "n_bridges_per_facility.png"))
    plt.clf()


def map_facilities_with_most_bridges():
    graph = BuildingRelationship.graph()
    bridges = list(nx.bridges(graph))
    facilities = list(nx.connected_components(graph))
    facilities_by_n_bridges = [
        (f, len([b for b in bridges if b[0] in f and b[1] in f]))
        for f in tqdm(facilities)
        if len(f) > 2
    ]
    facilities_by_n_bridges = sorted(
        facilities_by_n_bridges, key=lambda x: x[1], reverse=True
    )
    for i, (f, n) in enumerate(facilities_by_n_bridges):
        if n < 1:
            break
        print("Facility with most bridges: {}".format(n))
        map_ = map_building_set(f)
        map_.save(f"maps/facility_with_most_bridges/{i}_{n}.html")


def create_columbia_shapefile():
    facility_animal_types = {}
    for row in (
        FacilityAnimalType.select(FacilityAnimalType.facility_id, AnimalType.name)
        .join(AnimalType)
        .dicts()
    ):
        facility_animal_types[row["facility"]] = facility_animal_types.get(
            row["facility"], set()
        ) | {row["name"]}
    facility_animal_types = {k: list(v) for k, v in facility_animal_types.items()}

    facility_parcel_numbers = {}
    for row in (
        Facility.select(Facility.id, Parcel.numb, County.name)
        .join(Building)
        .join(Parcel)
        .join(County)
        .dicts()
    ):
        facility_parcel_numbers[row["id"]] = facility_parcel_numbers.get(
            row["id"], set()
        ) | {(row["numb"], row["name"])}
    facility_parcel_numbers = {k: list(v) for k, v in facility_parcel_numbers.items()}

    facilities_with_permits = {
        row["id"]
        for row in (
            Facility.select(Facility.id).join(FacilityPermittedLocation).dicts()
        )
    }
    facilities = (
        Facility.select(
            Facility.id,
            Facility.geometry,
            Facility.longitude,
            Facility.latitude,
            Facility.uuid,
            ConstructionAnnotation.construction_lower_bound.alias("conlowbnd"),
            ConstructionAnnotation.construction_upper_bound.alias("conupbnd"),
            ConstructionAnnotation.destruction_lower_bound.alias("deslowbnd"),
            ConstructionAnnotation.destruction_upper_bound.alias("desupbnd"),
            ConstructionAnnotation.significant_population_change.alias("sigpopchg"),
            ConstructionAnnotation.indoor_outdoor.alias("in/out"),
            ConstructionAnnotation.has_lagoon.alias("has_lagoon"),
        )
        .join(ConstructionAnnotation)
        .dicts()
    )
    gdf = gpd.GeoDataFrame(facilities)
    gdf["anmltypes"] = gdf["id"].apply(
        lambda x: json.dumps(facility_animal_types.get(x, []))
    )
    gdf["parcelnums"] = gdf["id"].apply(lambda x: facility_parcel_numbers.get(x, []))
    gdf["permit<1km"] = gdf["id"].apply(lambda x: x in facilities_with_permits)
    gdf.crs = "EPSG:4326"

    # shapefile has max 10 character column names
    gdf.rename(
        columns={
            "id": "facilityid",
        },
        inplace=True,
    )

    # check to see if any parcel nums have more counties
    for row in gdf["parcelnums"]:
        num_counties = len(set([p[1] for p in row]))
        if num_counties > 1:
            print(row)
    gdf["pn"] = gdf["parcelnums"]
    gdf["parcelnums"] = gdf["pn"].apply(lambda x: json.dumps([p[0] for p in x]))
    gdf["counties"] = gdf["pn"].apply(lambda x: json.dumps(list({p[1] for p in x})))
    gdf["uuid"] = gdf["uuid"].apply(lambda x: str(x))
    gdf.drop(columns=["pn"], inplace=True)

    # check to see if text cols < 255
    for col in gdf.columns:
        if col == "geometry":
            continue
        print(col, gdf[col].apply(lambda x: len(str(x))).max())

    gdf.to_file("outputs/columbia_facilities/columbia_facilities.shp")
    return gdf


def create_saskia_csv():
    import datetime

    facility_id_to_county = {
        row["id"]: County.geocode(lon=row["longitude"], lat=row["latitude"]).name
        for row in Facility.select(
            Facility.id, Facility.longitude, Facility.latitude
        ).dicts()
    }
    rows = (
        Facility.select(
            Facility.id.alias("facility_id"),
            Facility.uuid.alias("facility_uuid"),
            Facility.latitude,
            Facility.longitude,
            ConstructionAnnotation.id.alias("construction_annotation_id"),
            ConstructionAnnotation.construction_lower_bound.alias(
                "construction_lower_bound"
            ),
            ConstructionAnnotation.construction_upper_bound.alias(
                "construction_upper_bound"
            ),
            ConstructionAnnotation.destruction_lower_bound.alias(
                "destruction_lower_bound"
            ),
            ConstructionAnnotation.destruction_upper_bound.alias(
                "destruction_upper_bound"
            ),
            ConstructionAnnotation.significant_population_change.alias(
                "significant_population_change"
            ),
            ConstructionAnnotation.indoor_outdoor.alias("indoor_outdoor"),
            ConstructionAnnotation.has_lagoon.alias("has_lagoon"),
        )
        .join(ConstructionAnnotation)
        .dicts()
    )
    rows = list(rows)
    rows = [{"county": facility_id_to_county[r["facility_id"]], **r} for r in rows]
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    writer = csv.DictWriter(
        open(f"outputs/construction_dates_temp_{date}.csv", "w"),
        fieldnames=rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(rows)
    return rows


def get_adjacent_buildings_for_project(epsg, dist):
    points = gpd.GeoSeries(
        [
            shp.Point(lon, lat)
            for lon, lat, id_ in Building.select(
                Building.longitude, Building.latitude, Building.id
            )
            .order_by(Building.id)
            .tuples()
        ]
    )
    building_ids = sum(
        Building.select(Building.id).order_by(Building.id).tuples(),
        (),
    )
    points.crs = "EPSG:4326"
    points = points.to_crs(epsg)
    array = np.array([points.x, points.y]).T
    tree = sp.spatial.KDTree(array)
    distances, indices = tree.query(array, 500)

    return distances, indices

    bulk = {}
    for building_id, dists, idxs in zip(building_ids, distances, indices):
        close = dists < dist
        for dist, idx in zip(dists[close], idxs[close]):
            if building_id == building_ids[idx]:
                continue
            bulk[building_id] = bulk.get(building_id, set()) | {building_ids[idx]}

    return nx.from_dict_of_lists({k: frozenset(v) for k, v in bulk.items()})


def num_adjacent_buildings_to_eps_for_epsgs():
    df = pd.DataFrame()
    epsgs = ["EPSG:3071", "EPSG:32611", "EPSG:32610", "EPSG:3311"]
    building_ids = sum(
        Building.select(Building.id).order_by(Building.id).tuples(),
        (),
    )
    points = gpd.GeoSeries(
        [
            shp.Point(lon, lat)
            for lon, lat, id_ in Building.select(
                Building.longitude, Building.latitude, Building.id
            )
            .order_by(Building.id)
            .tuples()
        ]
    )
    epsg_indexes = {}
    for epsg in epsgs:
        p = points.copy()
        p.crs = "EPSG:4326"
        p = p.to_crs(epsg)
        array = np.array([p.x, p.y]).T
        tree = sp.spatial.KDTree(array)
        distances, indices = tree.query(array, 500)
        epsg_indexes[epsg] = (distances, indices)

    def get_adjacent_buildings_for_project(epsg, eps):
        distances, indices = epsg_indexes[epsg]
        bulk = {}
        for building_id, dists, idxs in zip(building_ids, distances, indices):
            close = dists < eps
            for dist, idx in zip(dists[close], idxs[close]):
                if building_id == building_ids[idx]:
                    continue
                bulk[building_id] = bulk.get(building_id, set()) | {building_ids[idx]}

        return nx.from_dict_of_lists({k: frozenset(v) for k, v in bulk.items()})

    df["eps"] = np.concatenate((np.arange(100, 1000, 100),) * len(epsgs))
    df["epsg"] = np.concatenate([np.repeat(epsg, 9) for epsg in epsgs])
    df["num_facilities"] = [
        len(
            list(nx.connected_components(get_adjacent_buildings_for_project(epsg, eps)))
        )
        for eps, epsg in tqdm(zip(df["eps"], df["epsg"]), total=len(df))
    ]
    # add dots
    df["jittered_eps"] = df["eps"]
    df.loc[df["epsg"] == "EPSG:32610", "jittered_eps"] += 10
    df.loc[df["epsg"] == "EPSG:32611", "jittered_eps"] -= 10
    sns.lineplot(x="eps", y="num_facilities", hue="epsg", data=df, legend=False)
    sns.scatterplot(x="jittered_eps", y="num_facilities", hue="epsg", data=df)
    # remove legend title entirely
    plt.legend(title="", fontsize="small")
    # remap labels
    labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
    label_map = {
        "EPSG:3071": "Wisconsin plane",
        "EPSG:32611": "UTM 11N",
        "EPSG:32610": "UTM 10N",
        "EPSG:3311": "California plane",
    }
    for i, label in enumerate(labels):
        labels[i] = label_map[label]
    plt.gca().legend(handles=plt.gca().get_legend().legendHandles, labels=labels)
    # set legend labels
    plt.title("Number of Facility Groups vs. DBSCAN max distance")
    plt.xlabel("Max Distance (m)")
    plt.ylabel("Number of Facility Groups")
    plt.savefig(os.path.join(FIG_PATH, f"num_adjacent_buildings_to_eps_for_epsgs.png"))
    plt.clf()


def num_mismatches_by_delta():
    epsg_base = "EPSG:3071"
    epsg_delta = "EPSG:3311"
    building_ids = sum(
        Building.select(Building.id).order_by(Building.id).tuples(),
        (),
    )
    points = gpd.GeoSeries(
        [
            shp.Point(lon, lat)
            for lon, lat, id_ in Building.select(
                Building.longitude, Building.latitude, Building.id
            )
            .order_by(Building.id)
            .tuples()
        ]
    )
    epsg_indexes = {}
    for epsg in [epsg_base, epsg_delta]:
        p = points.copy()
        p.crs = "EPSG:4326"
        p = p.to_crs(epsg)
        array = np.array([p.x, p.y]).T
        tree = sp.spatial.KDTree(array)
        distances, indices = tree.query(array, 500)
        epsg_indexes[epsg] = (distances, indices)

    df = pd.DataFrame()
    df["delta"] = np.arange(-200, 200, 10)
    mismatches = []
    for delta in tqdm(df["delta"]):
        building_groups = {}
        for epsg in (epsg_base, epsg_delta):
            distances, indices = epsg_indexes[epsg_base]
            bulk = {}
            for building_id, dists, idxs in zip(building_ids, distances, indices):
                close = dists < (250 + delta * (epsg_delta == epsg))
                for dist, idx in zip(dists[close], idxs[close]):
                    if building_id == building_ids[idx]:
                        continue
                    bulk[building_id] = bulk.get(building_id, set()) | {
                        building_ids[idx]
                    }
            graph = nx.from_dict_of_lists({k: frozenset(v) for k, v in bulk.items()})
            building_groups[epsg] = frozenset(
                [frozenset(s) for s in nx.connected_components(graph)]
            )
        mismatches.append(
            len(building_groups[epsg_base] ^ building_groups[epsg_delta]) / 2
        )
    df["mismatches"] = mismatches
    sns.lineplot(x="delta", y="mismatches", data=df)
    sns.scatterplot(x="delta", y="mismatches", data=df)
    plt.title("Number of Mismatches vs. Delta")
    plt.xlabel("Delta (m)")
    plt.ylabel("Number of Mismatches")
    plt.savefig(os.path.join(FIG_PATH, f"num_mismatches_by_delta.png"))
    plt.clf()


def num_mismatches_by_max_dist():
    epsg_base = "EPSG:3071"
    epsg_delta = "EPSG:3311"
    building_ids = sum(
        Building.select(Building.id).order_by(Building.id).tuples(),
        (),
    )
    points = gpd.GeoSeries(
        [
            shp.Point(lon, lat)
            for lon, lat, id_ in Building.select(
                Building.longitude, Building.latitude, Building.id
            )
            .order_by(Building.id)
            .tuples()
        ]
    )
    epsg_indexes = {}
    for epsg in [epsg_base, epsg_delta]:
        p = points.copy()
        p.crs = "EPSG:4326"
        p = p.to_crs(epsg)
        array = np.array([p.x, p.y]).T
        tree = sp.spatial.KDTree(array)
        distances, indices = tree.query(array, 500)
        epsg_indexes[epsg] = (distances, indices)

    df = pd.DataFrame()
    df["eps"] = np.arange(100, 1000, 50)
    mismatches = []
    for eps in tqdm(df["eps"]):
        building_groups = {}
        for epsg in (epsg_base, epsg_delta):
            distances, indices = epsg_indexes[epsg_base]
            bulk = {}
            for building_id, dists, idxs in zip(building_ids, distances, indices):
                close = dists < eps
                for dist, idx in zip(dists[close], idxs[close]):
                    if building_id == building_ids[idx]:
                        continue
                    bulk[building_id] = bulk.get(building_id, set()) | {
                        building_ids[idx]
                    }
            graph = nx.from_dict_of_lists({k: frozenset(v) for k, v in bulk.items()})
            building_groups[epsg] = frozenset(
                [frozenset(s) for s in nx.connected_components(graph)]
            )
        mismatches.append(
            len(building_groups[epsg_base] ^ building_groups[epsg_delta]) / 2
        )
    df["mismatches"] = mismatches
    sns.lineplot(x="eps", y="mismatches", data=df)
    sns.scatterplot(x="eps", y="mismatches", data=df)
    plt.title("Number of Mismatches vs. Epsilon")
    plt.xlabel("Epsilon (m)")
    plt.ylabel("Number of Mismatches")
    plt.savefig(os.path.join(FIG_PATH, f"num_mismatches_by_max_dist.png"))
    plt.clf()


def num_facilities_by_lone_building_distance():
    df = pd.DataFrame()
    df["distance"] = np.arange(0, 1000, 50)
    df["facilities"] = [
        list(BuildingRelationship.facilities(lone_building_distance=distance))
        for distance in tqdm(df["distance"], total=len(df))
    ]
    df["num_facilities"] = df["facilities"].apply(len)
    df["num_lone_buildings"] = df["facilities"].apply(
        lambda x: sum([len(f) == 1 for f in x])
    )

    sns.lineplot(x="distance", y="num_facilities", data=df, label="# facilities")
    sns.scatterplot(x="distance", y="num_facilities", data=df)
    sns.lineplot(
        x="distance", y="num_lone_buildings", data=df, label="# lone buildings"
    )
    sns.scatterplot(x="distance", y="num_lone_buildings", data=df)
    plt.legend(fontsize="small")
    plt.title("Number of Facilities vs. Lone Building Distance")
    plt.xlabel("Lone Building Distance (m)")
    plt.ylabel("Number of Facilities")
    plt.savefig(os.path.join(FIG_PATH, "num_facilities_by_lone_building_distance.png"))
    plt.clf()


def map_facilities_prior_destructions():
    facilities = list(
        Facility.select()
        .join(ConstructionAnnotation)
        .where(ConstructionAnnotation.destruction_upper_bound < 2017)
    )
    # map all facilities and save them to maps/
    os.makedirs("maps/facilities_prior_destructions", exist_ok=True)
    for i, facility in enumerate(facilities):
        gdf = facility.to_gdf()
        ax = gdf.plot(alpha=0.5, linewidth=0.5, edgecolor="black")
        ax.set_axis_off()
        naip.add_basemap(ax)

        plt.title(f"Facility {facility.id}")
        plt.savefig(f"maps/facilities_prior_destructions/{i}.png")


def facility_image_count_histogram():
    df = pd.DataFrame(
        list(
            Facility.select(
                Facility.id,
                pw.fn.COUNT(Building.image_id.distinct()).alias("n_images"),
            )
            .join(Building)
            .group_by(Facility.id)
            .dicts()
        )
    )
    sns.histplot(data=df, x="n_images", bins=range(0, 11), discrete=True)
    plt.title("Number of Images per Facility")
    plt.xlabel("Number of Images")
    plt.ylabel("Number of Facilities")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "facility_image_count_histogram.png"))
    plt.clf()


def bucket_type_count_histogram():
    df = pd.DataFrame(
        list(
            Facility.select(
                Facility.id,
                pw.fn.COUNT(Image.bucket.distinct()).alias("n_buckets"),
            )
            .join(Building)
            .join(Image)
            .group_by(Facility.id)
            .dicts()
        )
    )
    sns.histplot(data=df, x="n_buckets", bins=range(0, 4), discrete=True)
    plt.title("Number of Distinct Bucket Types per Facility")
    plt.xlabel("Number of Bucket Types")
    plt.ylabel("Number of Facilities")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "bucket_type_count_histogram.png"))
    plt.clf()


def compare_image_distribution_zero_one():
    facility_buckets = (
        Facility.select(
            Facility.id,
            pw.fn.MAX(Image.bucket).alias("bucket"),
        )
        .join(Building)
        .join(Image)
        .group_by(Facility.id, Image.id)
        .dicts()
    )
    df = pd.DataFrame(list(facility_buckets))

    # agg by id, group buckets
    df = df.groupby("id").agg(
        buckets=pd.NamedAgg(column="bucket", aggfunc=lambda x: tuple(sorted(x))),
        count=pd.NamedAgg(column="bucket", aggfunc="count"),
    )
    df = df.groupby("buckets").agg(count=pd.NamedAgg(column="count", aggfunc="sum"))
    df["buckets"] = df.index
    df = df[df["buckets"].apply(lambda x: all([b in ("0", "1") for b in x]))]

    # add labels to top of bars
    sns.barplot(x="buckets", y="count", data=df)
    for i, count in enumerate(df["count"]):
        plt.text(i, count, str(count), ha="center", va="bottom")
    plt.title("Number of Facilities by Bucket Mixture")
    plt.xlabel("Bucket Mixture")
    plt.ylabel("Number of Facilities")
    plt.ylim(0, 500)
    plt.xticks(rotation=90, ha="right", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "compare_image_distribution_zero_one.png"))
    plt.clf()


def unlabeled_adjacent_images():
    detections = Image.select(Image.geometry).where(Image.bucket > "1")

    query = (
        Image.select(
            pw.fn.COUNT(Image.id).alias("n_images"),
        )
        .join(
            detections,
            on=pw.fn.ST_Intersects(Image.geometry, detections.c.geometry),
        )
        .where(Image.label_status == "unlabeled")
        .count()
    )
    return query


def images_with_a_permit_query():
    _Image = Image.alias()
    query = (
        _Image.select(_Image.id, _Image.geometry, _Image.bucket, _Image.label_status)
        .join(
            PermittedLocation,
            on=pw.fn.ST_Contains(_Image.geometry, PermittedLocation.geometry),
        )
        .join(PermitPermittedLocation)
        .join(Permit)
        .group_by(_Image.id, _Image.geometry, _Image.bucket, _Image.label_status)
    )
    return query


def conf_interval_with_and_without_permit_strata():
    _Image = Image.alias()
    images_with_buildings = _Image.select(_Image.id).join(Building).distinct()
    pi = Image.alias()
    permit_images = (
        pi.select(pi.id)
        .join(
            PermittedLocation,
            on=pw.fn.ST_Contains(pi.geometry, PermittedLocation.geometry),
        )
        .distinct()
    )
    unlabeled_image = Image.label_status == "unlabeled"
    post_hoc_image = (Image.label_status == "permit") | (
        Image.label_status == "adjacent"
    )
    unsampled_image = unlabeled_image | post_hoc_image
    sampled_image = Image.label_status == "active learner"

    positive_image = images_with_buildings.c.id.is_null(False)
    sampled_positive_image = sampled_image & positive_image

    no_permit_query = (
        Image.select(
            Image.bucket,
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(unlabeled_image, 1)],
                    0,
                )
            ).alias("unlabeled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(unsampled_image, 1)],
                    0,
                )
            ).alias("unsampled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(sampled_image, 1)],
                    0,
                )
            ).alias("sampled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(post_hoc_image, 1)],
                    0,
                )
            ).alias("post_hoc_labeled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(sampled_positive_image, 1)],
                    0,
                )
            ).alias("sampled_positive"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [
                        (
                            positive_image,
                            1,
                        )
                    ],
                    0,
                )
            ).alias("positive"),
            pw.fn.COUNT("*").alias("total"),
        )
        .join(
            images_with_buildings,
            pw.JOIN.LEFT_OUTER,
            on=(Image.id == images_with_buildings.c.id),
        )
        .join(permit_images, pw.JOIN.LEFT_OUTER, on=(Image.id == permit_images.c.id))
        .where(Image.label_status != "removed")
        .group_by(Image.bucket)
        .order_by(Image.bucket)
    )
    no_permit_df = pd.DataFrame(list(no_permit_query.dicts()))

    permit_or_bucket = pw.Case(
        None,
        [(permit_images.c.id.is_null(False), "permit")],
        Image.bucket,
    )
    permit_query = (
        Image.select(
            permit_or_bucket.alias("bucket"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(unlabeled_image, 1)],
                    0,
                )
            ).alias("unlabeled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(unsampled_image, 1)],
                    0,
                )
            ).alias("unsampled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(sampled_image, 1)],
                    0,
                )
            ).alias("sampled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(post_hoc_image, 1)],
                    0,
                )
            ).alias("post_hoc_labeled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(sampled_positive_image, 1)],
                    0,
                )
            ).alias("sampled_positive"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [
                        (
                            positive_image,
                            1,
                        )
                    ],
                    0,
                )
            ).alias("positive"),
            pw.fn.COUNT("*").alias("total"),
        )
        .join(
            images_with_buildings,
            pw.JOIN.LEFT_OUTER,
            on=(Image.id == images_with_buildings.c.id),
        )
        .join(permit_images, pw.JOIN.LEFT_OUTER, on=(Image.id == permit_images.c.id))
        .where(Image.label_status != "removed")
        .group_by(permit_or_bucket)
        .order_by(permit_or_bucket)
    )
    permit_df = pd.DataFrame(list(permit_query.dicts()))

    for df, label in zip((no_permit_df, permit_df), ("no permit", "permit")):
        df["sampled_prevalence"] = df["sampled_positive"] / df["sampled"]
        df["sampled_prevalence_lower"] = df.apply(
            lambda x: proportion_confint(
                x["sampled_positive"], x["sampled"], method="beta"
            )[0],
            axis=1,
        )
        df["sampled_prevalence_upper"] = df.apply(
            lambda x: proportion_confint(
                x["sampled_positive"], x["sampled"], method="beta"
            )[1],
            axis=1,
        )
        df["population_estimate_lower"] = (
            df["sampled_prevalence_lower"] * df["unlabeled"] + df["positive"]
        )
        df["population_estimate_upper"] = (
            df["sampled_prevalence_upper"] * df["unlabeled"] + df["positive"]
        )
        df["population_estimate"] = (
            df["sampled_prevalence"] * df["unlabeled"] + df["positive"]
        )
        print(label)
        print(df)

    return no_permit_df, permit_df


def flagged_images_id_uuid_coord():
    label = pw.fn.JSON_ARRAY_ELEMENTS(
        RawCloudFactoryImageAnnotation.json["annotations"]
    )["label"]
    flagged_images = (
        RawCloudFactoryImageAnnotation.select(
            label.alias("label"),
            RawCloudFactoryImageAnnotation.image_id,
        )
        .where(label == "flag")
        .distinct()
    )
    query = (
        Image.select(
            Image.id,
            Image.name,
            Image.geometry,
        )
        .join(flagged_images, on=(Image.id == flagged_images.c.image_id))
        .dicts()
    )
    return list(query)


def facilities_without_construction_annotation():
    return list(
        Facility.select(
            Facility.id, Facility.uuid, Facility.latitude, Facility.longitude
        )
        .join(ConstructionAnnotation, pw.JOIN.LEFT_OUTER)
        .where(ConstructionAnnotation.id.is_null())
        .dicts()
    )


def image_bucket_clusters():
    other_image = Image.alias()
    id_bucket_geom = sorted(
        list(
            Image.select(
                Image.id,
                Image.bucket,
                Image.geometry,
            )
            .join(Building)
            .dicts()
            .distinct(Image.id)
        ),
        key=lambda x: x["id"],
    )
    id_to_bucket = {row["id"]: row["bucket"] for row in id_bucket_geom}
    graph = cache.get("image_cluster_graph")
    if not graph:
        geoms = [row["geometry"] for row in id_bucket_geom]
        tree = shp.STRtree(geoms)
        geom_1, geom_2 = tree.query(geoms, predicate="intersects")
        images_1 = [id_bucket_geom[i]["id"] for i in geom_1]
        images_2 = [id_bucket_geom[i]["id"] for i in geom_2]
        edgelist = [(i, j) for i, j in zip(images_1, images_2) if i < j]
        graph = nx.from_edgelist(edgelist)
        cache.set("image_cluster_graph", graph)
    clusters = list(nx.connected_components(graph))

    bucket_cluster_count = {}
    for c in clusters:
        bucket_cluster = tuple(sorted([id_to_bucket[i] for i in c]))
        bucket_cluster_count[bucket_cluster] = (
            bucket_cluster_count.get(bucket_cluster, 0) + 1
        )
    return bucket_cluster_count


def fractional_facility_counts():
    db.execute_sql("set work_mem to '5GB'")
    adjacent_images = Image.alias()

    facility_to_n_images = {
        row["id"]: row["n_images"]
        for row in (
            Facility.select(
                Facility.id,
                pw.fn.COUNT(Image.id).alias("n_images"),
            )
            .join(Building)
            .join(Image)
            .group_by(Facility.id)
            .dicts()
        )
    }
    zero_images = cache.get("zero_images")
    one_images = cache.get("one_images")

    if True:  # not zero_images or not one_images:
        zero_images = (
            Image.select()
            # .join(ImageAdjacency, on=(Image.id == ImageAdjacency.image_id))
            # .join(adjacent_images, on=(ImageAdjacency.adjacent_image_id == adjacent_images.id))
            .join(Building, on=(Building.image_id == Image.id))
            .where((Image.bucket == "0") & Building.cafo)
            .group_by(Image.id)
            # .having(pw.fn.MAX(adjacent_images.bucket) <= "3")
        )
        one_images = (
            Image.select()
            # .join(ImageAdjacency, on=(Image.id == ImageAdjacency.image_id))
            # .join(adjacent_images, on=(ImageAdjacency.adjacent_image_id == adjacent_images.id))
            .join(Building, on=(Image.id == Building.image_id))
            .where((Image.bucket == "1") & Building.cafo)
            .group_by(Image.id)
            # .having(pw.fn.MAX(adjacent_images.bucket) <= "3")
        )

        facilities = Facility.select()
        buildings = Building.select()

        zero_images = pw.prefetch(zero_images, buildings, facilities)
        one_images = pw.prefetch(one_images, buildings, facilities)

        zero_images = [
            {
                "complete_facilities": len(
                    [
                        f
                        for f in zero_image.facilities()
                        if facility_to_n_images[f.id] == 1
                    ]
                ),
                "half_facilities": len(
                    [
                        f
                        for f in zero_image.facilities()
                        if facility_to_n_images[f.id] == 2
                    ]
                ),
                "third_facilities": len(
                    [
                        f
                        for f in zero_image.facilities()
                        if facility_to_n_images[f.id] == 3
                    ]
                ),
                "quarter_facilities": len(
                    [
                        f
                        for f in zero_image.facilities()
                        if facility_to_n_images[f.id] >= 4
                    ]
                ),
            }
            for zero_image in zero_images
        ]
        one_images = [
            {
                "complete_facilities": len(
                    [
                        f
                        for f in one_image.facilities()
                        if facility_to_n_images[f.id] == 1
                    ]
                ),
                "half_facilities": len(
                    [
                        f
                        for f in one_image.facilities()
                        if facility_to_n_images[f.id] == 2
                    ]
                ),
                "third_facilities": len(
                    [
                        f
                        for f in one_image.facilities()
                        if facility_to_n_images[f.id] == 3
                    ]
                ),
                "quarter_facilities": len(
                    [
                        f
                        for f in one_image.facilities()
                        if facility_to_n_images[f.id] == 4
                    ]
                ),
            }
            for one_image in one_images
        ]
        cache.set("zero_images", zero_images)
        cache.set("one_images", one_images)

    zero_images = pd.DataFrame(zero_images)
    one_images = pd.DataFrame(one_images)

    df = pd.read_csv("paper/tables/csv/recall.csv")
    df["Missing Lower"] = df["Population Estimate Lower"] - df["Positive"]
    df["Missing Upper"] = df["Population Estimate Upper"] - df["Positive"]
    df["Missing"] = df["Population Estimate"] - df["Positive"]

    missing_zero = int(df.loc[df["Bucket"] == "0", "Missing"].values[0])
    missing_one = int(df.loc[df["Bucket"] == "1", "Missing"].values[0])

    estimates = []
    bootstrap_iterations = 1000
    for i in range(bootstrap_iterations):
        zero_sample = zero_images.sample(missing_zero, replace=True)
        one_sample = one_images.sample(missing_one, replace=True)
        totals = zero_sample.sum() + one_sample.sum()
        total = (
            totals["complete_facilities"]
            + totals["half_facilities"] // 2
            + totals["third_facilities"] // 3
            + totals["quarter_facilities"] // 4
        )
        estimates.append(total)
    estimates = np.array(estimates)
    median = np.median(estimates)
    n_facilities = len(facility_to_n_images)

    return n_facilities / (n_facilities + median)


def ex_ante_adjacents():
    AdjacentImage = Image.alias()
    image_names = sum(
        Image.select(
            AdjacentImage.name,
        )
        .join(Building)
        .join(
            ImageAdjacency,
            on=(Image.id == ImageAdjacency.image_id),
        )
        .join(
            AdjacentImage,
            on=(ImageAdjacency.adjacent_image_id == AdjacentImage.id),
        )
        .where(
            (Image.bucket == "ex ante permit")
            & (AdjacentImage.label_status == "unlabeled")
        )
        .distinct()
        .tuples(),
        (),
    )
    return image_names


@cache.memoize()
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


def _clean_facility_geometry(facility):
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


def facility_geojson(fname=None):
    facilities = Facility.select()
    facilities_with_buildings = pw.prefetch(
        Facility.select(), Building.select().where(Building.cafo), Parcel.select()
    )

    records = []
    for facility in facilities_with_buildings:
        if not all([isinstance(b.geometry, shp.Polygon) for b in facility.buildings]):
            raise ValueError(f"Facility {facility.id} has non-polygon geometry")
        geometry = _clean_facility_geometry(facility)
        owners = set(
            [b.parcel.owner for b in facility.buildings if b.parcel and b.parcel.owner]
        )
        records.append(
            {
                "geometry": geometry,
                "latitude": facility.latitude,
                "longitude": facility.longitude,
                "parcel_owners": "; ".join(owners),
                "id": facility.id,
                "uuid": str(facility.uuid),
            }
        )
    df = gpd.GeoDataFrame(records)
    df["uuid"] = df["uuid"].apply(str)
    date = datetime.now().strftime("%Y-%m-%d")
    df = df.set_crs("EPSG:4326")
    if not fname:
        fname = f"outputs/facilities_{date}.geojson"
    df.to_file(fname, driver="GeoJSON")
    print(f"Saved to {fname}")
    return df


def facility_centroids_geojson(fname=None):
    facilities = Facility.select()
    facilities_with_buildings = pw.prefetch(
        Facility.select(), Building.select().where(Building.cafo), Parcel.select()
    )

    records = []
    for facility in facilities_with_buildings:
        if not all([isinstance(b.geometry, shp.Polygon) for b in facility.buildings]):
            raise ValueError(f"Facility {facility.id} has non-polygon geometry")
        geometry = _clean_facility_geometry(facility)
        owners = set(
            [b.parcel.owner for b in facility.buildings if b.parcel and b.parcel.owner]
        )
        records.append(
            {
                "geometry": geometry.centroid,
                "latitude": facility.latitude,
                "longitude": facility.longitude,
                "parcel_owners": "; ".join(owners),
                "id": facility.id,
                "uuid": str(facility.uuid),
            }
        )
    df = gpd.GeoDataFrame(records)
    df["uuid"] = df["uuid"].apply(str)
    date = datetime.now().strftime("%Y-%m-%d")
    df = df.set_crs("EPSG:4326")
    if not fname:
        fname = f"outputs/facilities_centroids_{date}.geojson"
    df.to_file(fname, driver="GeoJSON")
    print(f"Saved to {fname}")
    return df


def permits_geojson(fname=None):
    permits = pw.prefetch(
        Permit.select(), PermitPermittedLocation.select(), PermittedLocation.select()
    )
    gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": p.location().geometry,
                "facility_name": p.facility_name,
                "facility_address": p.facility_address,
                "permitted_population": p.permitted_population,
                "regulatory_status": p.regulatory_measure_status,
                "agency_name": p.agency_name,
                "agency_address": p.agency_address,
                "wdid": p.wdid,
            }
            for p in permits
            if p.location()
        ]
    )
    gdf.to_file("outputs/permits.geojson", driver="GeoJSON")


def facility_level_population_estimate():
    import multiplicity

    facilities = {
        f[0]: multiplicity.Individual(num_reported_by=f[1])
        for f in Facility.select(Facility.id, pw.fn.COUNT(Image.id.distinct()))
        .join(Building)
        .join(Image)
        .group_by(Facility.id)
        .tuples()
    }
    images = {}
    for id_ in (
        Image.select(Image.id)
        .where(~Image.label_status.in_(("unlabeled", "removed")))
        .tuples()
    ):
        images[id_[0]] = multiplicity.Source(reported_individuals=set())

    for id_ in (
        Image.select(Image.id, Facility.id)
        .join(Building)
        .join(Facility)
        .where(~Image.label_status.in_(("unlabeled", "removed")))
        .tuples()
    ):
        images[id_[0]].reported_individuals.add(facilities[id_[1]])

    samples = {
        i[0]: multiplicity.Stratum(
            total_number_of_sources_in_stratum=i[1],
            sampled_sources=[],
            unsampled_labeled_sources=[],
        )
        for i in Image.select(Image.bucket, pw.fn.COUNT("*"))
        .group_by(Image.bucket)
        .tuples()
        if i[0] is not None
    }
    for bucket, id_, label_status in (
        Image.select(Image.bucket, Image.id, Image.label_status)
        .where(~Image.label_status.in_(("unlabeled", "removed")))
        .tuples()
    ):
        if label_status == "active learner":
            samples[bucket].sampled_sources.append(images[id_])
        else:
            samples[bucket].unsampled_labeled_sources.append(images[id_])
    # import ipdb; ipdb.set_trace()
    survey = multiplicity.Survey(strata=list(samples.values()))
    print(survey.unbiased_population_estimate(bootstrap_iters=200))
    return survey


def histogram_of_weighted_sources():
    survey = facility_level_population_estimate()
    zero_source_weights = [
        source.weighted_num_reported() for source in survey.strata[0].sampled_sources
    ]
    full_zero_distribution = [
        source.weighted_num_reported()
        for source in survey.strata[0].sampled_sources
        + survey.strata[0].unsampled_labeled_sources
    ]

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "num_reported": zero_source_weights,
                    "label": "sampled",
                }
            ),
            pd.DataFrame(
                {
                    "num_reported": full_zero_distribution,
                    "label": "full",
                }
            ),
        ]
    )
    sns.histplot(
        data=df,
        x="num_reported",
        hue="label",
        bins=np.arange(0, 2.25, 0.25),
        log_scale=(False, True),
    )
    plt.title("Histogram of Weighted Sources")
    plt.xlabel("Weighted Number of Reported Individuals")
    plt.ylabel("Number of Sources")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "weighted_sources_histogram.png"))


def unlabeled_adjacents():
    AdjacentImage = Image.alias()
    adjacent_image_ids = (
        Image.select(
            AdjacentImage.id,
        )
        .join(Building)
        .join(
            ImageAdjacency,
            on=(Image.id == ImageAdjacency.image_id),
        )
        .join(
            AdjacentImage,
            on=(ImageAdjacency.adjacent_image_id == AdjacentImage.id),
        )
        .where(
            # (Image.bucket == "ex ante permit")
            (AdjacentImage.label_status == "unlabeled")
            & (Image.label_status != "adjacent")
        )
        .distinct()
        .tuples()
    )
    return list(Image.select().where(Image.id.in_(adjacent_image_ids)))


def buildings_without_parcels_csv():
    output = open("outputs/buildings_without_parcels.csv", "w")
    buildings = list(
        Building.select(Building.id, Building.latitude, Building.longitude)
        .where(Building.parcel_id.is_null() & (Building.cafo))
        .dicts()
    )
    writer = csv.DictWriter(output, fieldnames=["id", "latitude", "longitude"])
    writer.writeheader()
    for building in buildings:
        writer.writerow(
            {
                "id": building["id"],
                "latitude": building["latitude"],
                "longitude": building["longitude"],
            }
        )
    output.close()


def facilities_without_buildings():
    return list(
        Facility.select()
        .join(Building, pw.JOIN.LEFT_OUTER)
        .where(Building.id.is_null())
        .dicts()
    )


def old_recall():
    unlabeled_image = Image.label_status == "unlabeled"
    labeled_image = (Image.label_status != "unlabeled") & (
        Image.label_status != "removed"
    )
    PositiveImage = Image.alias()
    # this is a workaround for a peewee bug
    # see https://github.com/coleifer/peewee/issues/2873
    subquery = pw.NodeList(
        (
            pw.SQL("("),
            PositiveImage.select(PositiveImage.id)
            .join(Building)
            .where(Building.cafo)
            .distinct(),
            pw.SQL(")"),
        )
    )

    positive_image = Image.id.in_(subquery)

    query = (
        Image.select(
            Image.bucket,
            CountyGroup.name.alias("county_group"),
            pw.fn.COUNT("*").alias("n_images"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(unlabeled_image, 1)],
                    0,
                )
            ).alias("unlabeled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(labeled_image, 1)],
                    0,
                )
            ).alias("labeled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(positive_image, 1)],
                    0,
                )
            ).alias("positive"),
        )
        .join(County)
        .join(CountyGroup)
        .group_by(Image.bucket, CountyGroup.name)
        .where(Image.label_status != "adjacent")
        .dicts()
    )
    return pd.DataFrame(query)


def recall():
    unlabeled_image = Image.label_status == "unlabeled"
    labeled_image = (Image.label_status != "unlabeled") & (
        Image.label_status != "removed"
    )
    PositiveImage = Image.alias()
    # this is a workaround for a peewee bug
    # see https://github.com/coleifer/peewee/issues/2873
    subquery = pw.NodeList(
        (
            pw.SQL("("),
            PositiveImage.select(PositiveImage.id)
            .join(Building)
            .where(Building.cafo)
            .distinct(),
            pw.SQL(")"),
        )
    )

    positive_image = Image.id.in_(subquery)

    query = (
        Image.select(
            Image.stratum,
            pw.fn.COUNT("*").alias("n_images"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(unlabeled_image, 1)],
                    0,
                )
            ).alias("unlabeled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(labeled_image, 1)],
                    0,
                )
            ).alias("labeled"),
            pw.fn.SUM(
                pw.Case(
                    None,
                    [(positive_image, 1)],
                    0,
                )
            ).alias("positive"),
        )
        .group_by(Image.stratum)
        .dicts()
    )
    df = pd.DataFrame(query)
    df["prevalence"] = df["positive"] / df["labeled"]
    df["prevalence_lower"] = df.apply(
        lambda x: proportion_confint(x["positive"], x["labeled"], method="beta")[0],
        axis=1,
    )
    df["prevalence_upper"] = df.apply(
        lambda x: proportion_confint(x["positive"], x["labeled"], method="beta")[1],
        axis=1,
    )
    df["population_estimate_lower"] = (
        df["prevalence_lower"] * df["unlabeled"] + df["positive"]
    )
    df["population_estimate_upper"] = (
        df["prevalence_upper"] * df["unlabeled"] + df["positive"]
    )
    df["population_estimate"] = df["prevalence"] * df["unlabeled"] + df["positive"]
    return df


def facilities_per_image():
    return (
        Image.select().join(Building).distinct(Image.id).count()
    ) / Facility.select().count()


def columbia_geojson():
    query = pw.prefetch(
        Facility.select(),
        Permit.select(),
        Building.select(),
        Parcel.select(),
        County.select(County.id, County.name),
        FacilityPermittedLocation.select(),
        PermittedLocation.select(),
        FacilityAnimalType.select(),
        AnimalType.select(),
        ConstructionAnnotation.select(),
    )
    features = [facility.to_geojson_feature() for facility in tqdm(query)]
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def write_columbia_geojson():
    with open(
        f"outputs/columbia_{datetime.now().strftime('%Y-%m-%d')}.geojson", "w"
    ) as f:
        json.dump(columbia_geojson(), f)


def permits_with_disparate_locations(distance=100):
    permits = (
        Permit.select(
            Permit.id, PermittedLocation.latitude, PermittedLocation.longitude
        )
        .join(PermitPermittedLocation)
        .join(PermittedLocation)
        .distinct()
        .dicts()
    )
    permits = [
        {"id": p["id"], "geometry": shp.Point(p["longitude"], p["latitude"])}
        for p in permits
    ]
    gdf = gpd.GeoDataFrame(permits)
    gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:3311")
    permits = {}
    for idx, row in gdf.iterrows():
        permits[row["id"]] = permits.get(row["id"], []) + [row["geometry"]]
    disparate_permits = []
    for permit, permitted_locations in permits.items():
        if len(permitted_locations) <= 1:
            continue
        if len(permitted_locations) == 2:
            if permitted_locations[0].distance(permitted_locations[1]) > distance:
                disparate_permits.append(permit)
        if len(permitted_locations) > 2:
            raise ValueError("More than 2 permitted locations")
    return disparate_permits


def facilities_with_only_permit_animal_types():
    """how many facilities only have permit animal types?"""
    facilities = (
        Facility.select(Facility.id)
        .join(FacilityAnimalType)
        .group_by(Facility.id)
        .having(
            (pw.fn.COUNT(FacilityAnimalType.id) == 1)
            & (pw.fn.MAX(FacilityAnimalType.label_source) == "permit")
        )
        .dicts()
    )
    return {f["id"] for f in facilities}


def facility_permit_distance_matches(facility_permitted_location_conditions):
    fpls = (
        FacilityPermittedLocation.select(
            FacilityPermittedLocation.facility_id.alias("facility_id"),
            PermitPermittedLocation.permit_id.alias("permit_id"),
        )
        .join(PermittedLocation)
        .join(PermitPermittedLocation)
        .where(facility_permitted_location_conditions)
        .dicts()
    )
    facilities = {}
    for fpl in fpls:
        facilities[fpl["facility_id"]] = facilities.get(fpl["facility_id"], set()) | {
            fpl["permit_id"]
        }
    return facilities


def facility_permit_parcel_matches():
    facility_permits = (
        Building.select(
            Building.facility_id.alias("facility_id"),
            Permit.id.alias("permit_id"),
        )
        .join(Permit, on=(Building.parcel_id == Permit.parcel_id))
        .dicts()
    )
    facilities = {}
    for fp in facility_permits:
        facilities[fp["facility_id"]] = facilities.get(fp["facility_id"], set()) | {
            fp["permit_id"]
        }
    return facilities


def _conjunction_dict_of_sets(dos1, dos2):
    dos = dos1.copy()
    keys = set(dos1.keys()) | set(dos2.keys())
    return {k: dos1.get(k, set()) & dos2.get(k, set()) for k in keys}


def _disjunction_dict_of_sets(dos1, dos2):
    dos = dos1.copy()
    keys = set(dos1.keys()) | set(dos2.keys())
    return {k: dos1.get(k, set()) | dos2.get(k, set()) for k in keys}


def _invert_dict_of_sets(dos):
    sod = {}
    for k, v in dos.items():
        for i in v:
            sod[i] = sod.get(i, set()) | {k}
    return sod


def _remove_duplicate_entries(dos):
    """In a dict of sets, remove any set entries that are in another set entry"""
    sod = _invert_dict_of_sets(dos)
    dos_output = {}
    for s, d_set in sod.items():
        if len(d_set) == 1:
            d = d_set.pop()
            dos_output[d] = dos_output.get(d, set()) | {s}
    return dos_output


def _remove_empty_entries(dos):
    return {k: v for k, v in dos.items() if v}


def permit_sensitivity_analysis():
    plt.figure(figsize=(6, 4))

    cow_permit_ids = {
        p.id
        for p in Permit.select(Permit.id).where(Permit.data["Program"] == "ANIWSTCOWS")
    }
    cow_permit_matches = lambda matches: _remove_empty_entries(
        {
            f: {p for p in permits if p in cow_permit_ids}
            for f, permits in matches.items()
        }
    )
    facilities_without_human_labels = (
        facilities_with_only_permit_animal_types() | facilities_without_animal_types()
    )
    n_hand_label = lambda matches: len(
        facilities_without_human_labels - set(cow_permit_matches(matches).keys())
    )

    matching_distances = range(0, 1001, 50)

    permit_data_filter = PermitPermittedLocation.source == "permit data"
    geocoding_filter = PermitPermittedLocation.source == "address geocoding"
    distance_filter = lambda d: FacilityPermittedLocation.distance < d

    parcel_matches = facility_permit_parcel_matches()

    data = []
    for distance in matching_distances:
        # permit data only
        location_type = {
            "permit data": facility_permit_distance_matches(
                permit_data_filter & distance_filter(distance)
            ),
            "geocoding": facility_permit_distance_matches(
                geocoding_filter & distance_filter(distance)
            ),
        }
        location_type["both"] = _conjunction_dict_of_sets(
            location_type["permit data"], location_type["geocoding"]
        )
        location_type["either"] = _disjunction_dict_of_sets(
            location_type["permit data"], location_type["geocoding"]
        )
        for location_type_name, location_type_matches in location_type.items():
            row = {
                "distance": distance,
                "location_type": location_type_name,
            }

            matches = location_type_matches

            distance_only_matches = _remove_empty_entries(
                _remove_duplicate_entries(matches)
            )
            distance_only_row = {
                "matching_method": "distance only",
                "n_clean_matches": len(distance_only_matches),
                "n_hand_label": n_hand_label(distance_only_matches),
            } | row
            data.append(distance_only_row)

            parcel_or_distance_matches = _remove_empty_entries(
                _remove_duplicate_entries(
                    _disjunction_dict_of_sets(matches, parcel_matches)
                )
            )
            parcel_or_distance_row = {
                "matching_method": "parcel + distance",
                "n_clean_matches": len(parcel_or_distance_matches),
                "n_hand_label": n_hand_label(parcel_or_distance_matches),
            } | row
            data.append(parcel_or_distance_row)

            parcel_then_distance_matches = _disjunction_dict_of_sets(
                parcel_matches,
                _remove_empty_entries(_remove_duplicate_entries(matches)),
            )
            parcel_then_distance_row = {
                "matching_method": "parcel then distance",
                "n_clean_matches": len(parcel_then_distance_matches),
                "n_hand_label": n_hand_label(parcel_then_distance_matches),
            } | row
            data.append(parcel_then_distance_row)

    df = pd.DataFrame(data)
    # make lineplot
    sns.lineplot(
        data=df,
        x="distance",
        y="n_clean_matches",
        hue="location_type",
        style="matching_method",
    )
    plt.title("Number of Clean Matches by Matching Distance")
    plt.xlabel("Matching Distance (m)")
    plt.legend(title=None, loc="upper right", fontsize="small")

    plt.ylabel("Number of Clean Matches")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "clean_matches_by_distance.png"))

    # make lineplot
    plt.close()
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=df,
        x="distance",
        y="n_hand_label",
        hue="location_type",
        style="matching_method",
    )
    plt.title("Number of Facilities Needing Hand Labeling by Matching Distance")
    plt.xlabel("Matching Distance (m)")
    plt.ylabel("Number of Facilities Needing Hand Labeling")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "hand_label_by_distance.png"))
    return df


def parcel_then_distance_matches(distance=200, cow_only=False) -> dict[int, set[int]]:
    """
    Returns a dictionary of facility ids to sets of permit ids that match
    based on the parcel and then the distance between the facility and the permit
    """
    permit_data_filter = PermitPermittedLocation.source == "permit data"
    geocoding_filter = PermitPermittedLocation.source == "address geocoding"
    distance_filter = lambda d: FacilityPermittedLocation.distance < d

    parcel_matches = facility_permit_parcel_matches()
    permit_matches = facility_permit_distance_matches(
        permit_data_filter & distance_filter(200)
    )
    geocoded_matches = facility_permit_distance_matches(
        geocoding_filter & distance_filter(200)
    )

    distance_matches = _conjunction_dict_of_sets(permit_matches, geocoded_matches)

    matches = _disjunction_dict_of_sets(
        parcel_matches,
        _remove_empty_entries(_remove_duplicate_entries(distance_matches)),
    )
    cow_permit_ids = {
        p.id
        for p in Permit.select(Permit.id).where(Permit.data["Program"] == "ANIWSTCOWS")
    }
    cow_permit_matches = lambda matches: _remove_empty_entries(
        {
            f: {p for p in permits if p in cow_permit_ids}
            for f, permits in matches.items()
        }
    )
    return matches if not cow_only else cow_permit_matches(matches)


def distinct_animal_types():
    facilities_with_animal_types = pw.prefetch(
        Facility.select(),
        FacilityAnimalType.select(),
        AnimalType.select(),
    )
    animal_types = [
        ", ".join(sorted({at.name for at in f.animal_types}))
        for f in facilities_with_animal_types
    ]
    return pd.Series(animal_types).value_counts()


def facilities_without_buildings():
    return list(
        Facility.select()
        .join(Building, pw.JOIN.LEFT_OUTER)
        .where(Building.id.is_null())
    )


if __name__ == "__main__":
    pass
