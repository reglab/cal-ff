import io
import itertools
import random
from glob import glob

import diskcache as dc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peewee as pw
import recall
import schemdraw
import seaborn as sns
from models import *
from PIL import Image
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

import naip

sns.set_palette("Set2")
sns.set_theme("paper", style="white", font="Times New Roman")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (3, 2.5)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}\usepackage{amsmath}"

plot_methods = []
table_methods = []

cache = dc.Cache("cache")

BOOTSTRAP_ITERS = 5000


def data_hash():
    return hash(
        (tuple(Facility.select(Facility.uuid).tuples()),)
        + tuple((m.select().count() for m in models()))
    )


if "data_hash" in cache:
    if data_hash() != cache["data_hash"]:
        cache.clear()
else:
    cache["data_hash"] = data_hash()


def plot_method(method):
    plot_methods.append(method)
    return method


def table_method(method):
    table_methods.append(method)
    return method


def data_method(method):
    def wrapper(*args, **kwargs):
        key = method.__name__
        if key in cache:
            return cache[key]
        else:
            result = method(*args, **kwargs)
            cache[key] = result
            return result

    return method


@data_method
def facility_counts_by_county():
    permitted_facilities = (
        Facility.select(
            Facility.id,
        )
        .join(
            FacilityPermittedLocation,
            pw.JOIN.LEFT_OUTER,
        )
        .where(
            FacilityPermittedLocation.id.is_null(False),
        )
        .distinct()
        .cte("permitted_facilities")
    )
    facility_to_county = {
        facility.id: County.geocode(lon=facility.longitude, lat=facility.latitude).id
        for facility in Facility.select()
    }

    class FacilityCounty(pw.Model):
        facility = pw.ForeignKeyField(Facility, primary_key=True)
        county = pw.ForeignKeyField(County)

        class Meta:
            database = db
            table_name = "facility_county"

    FacilityCounty.create_table()
    FacilityCounty.insert_many(
        [
            {"facility": facility, "county": county}
            for facility, county in facility_to_county.items()
        ]
    ).execute()
    counts = (
        FacilityCounty.select(
            County.name.alias("county"),
            pw.fn.SUM(
                pw.Case(None, ((permitted_facilities.c.id.is_null(False), 1),), 0)
            ).alias("permitted_count"),
            pw.fn.SUM(
                pw.Case(None, ((permitted_facilities.c.id.is_null(True), 1),), 0)
            ).alias("unpermitted_count"),
            pw.fn.COUNT(FacilityCounty.facility).alias("total_count"),
        )
        .join(
            County,
            pw.JOIN.LEFT_OUTER,
        )
        .join(
            permitted_facilities,
            pw.JOIN.LEFT_OUTER,
            on=(permitted_facilities.c.id == FacilityCounty.facility),
        )
        .group_by(
            County.name,
        )
        .with_cte(permitted_facilities)
        .dicts()
    )
    counts = pd.DataFrame(counts)
    FacilityCounty.drop_table()
    sorted_counts = counts.sort_values("permitted_count", ascending=False)
    return sorted_counts


@data_method
def constructions_and_destructions_by_year():
    facilities = pd.DataFrame(
        {
            "year": range(1985, 2024),
        }
    )
    facilities["started_constructions"] = facilities["year"].apply(
        lambda x: Facility.select()
        .join(ConstructionAnnotation)
        .where(
            ConstructionAnnotation.construction_lower_bound == x,
        )
        .count()
    )
    facilities["completed_constructions"] = facilities["year"].apply(
        lambda x: Facility.select()
        .join(ConstructionAnnotation)
        .where(
            (ConstructionAnnotation.construction_upper_bound == x)
            & (ConstructionAnnotation.construction_lower_bound.is_null(False))
        )
        .count()
    )
    facilities["started_destructions"] = facilities["year"].apply(
        lambda x: Facility.select()
        .join(ConstructionAnnotation)
        .where(
            ConstructionAnnotation.destruction_lower_bound == x,
        )
        .count()
    )
    facilities["completed_destructions"] = facilities["year"].apply(
        lambda x: Facility.select()
        .join(ConstructionAnnotation)
        .where(
            (ConstructionAnnotation.destruction_upper_bound == x)
            & (ConstructionAnnotation.destruction_lower_bound.is_null(False))
        )
        .count()
    )
    # flatten the dataframe
    facilities = facilities.melt(
        id_vars=["year"],
        var_name="event",
        value_name="count",
    )
    return facilities


@table_method
def permitted_by_animal_type():
    permitted_facilities = (
        Facility.select(
            Facility.id,
        )
        .join(
            FacilityPermittedLocation,
            pw.JOIN.LEFT_OUTER,
        )
        .where(
            FacilityPermittedLocation.id.is_null(False),
        )
        .distinct()
        .cte("permitted_facilities")
    )
    counts = (
        Facility.select(
            AnimalType.name.alias("animal_type"),
            pw.fn.SUM(
                pw.Case(None, ((permitted_facilities.c.id.is_null(False), 1),), 0)
            ).alias("permitted_count"),
            pw.fn.SUM(
                pw.Case(None, ((permitted_facilities.c.id.is_null(True), 1),), 0)
            ).alias("unpermitted_count"),
            pw.fn.COUNT(FacilityAnimalType.id).alias("total_count"),
        )
        .join(
            FacilityAnimalType,
            pw.JOIN.LEFT_OUTER,
        )
        .join(
            AnimalType,
            pw.JOIN.LEFT_OUTER,
        )
        .join(
            permitted_facilities,
            pw.JOIN.LEFT_OUTER,
            on=(permitted_facilities.c.id == Facility.id),
        )
        .group_by(
            AnimalType.name,
        )
        .with_cte(permitted_facilities)
        .dicts()
    )
    counts = pd.DataFrame(counts)
    sorted_counts = counts.sort_values("permitted_count", ascending=False)
    # add a total row to the bottom
    total_row = pd.Series(
        {
            "animal_type": "Total",
            "permitted_count": sorted_counts["permitted_count"].sum(),
            "unpermitted_count": sorted_counts["unpermitted_count"].sum(),
            "total_count": sorted_counts["total_count"].sum(),
        }
    )
    sorted_counts = sorted_counts.append(total_row, ignore_index=True)

    # rename columns
    sorted_counts = sorted_counts.rename(
        columns={
            "animal_type": "Animal Type",
            "permitted_count": "Permit <1km",
            "unpermitted_count": "No Permit <1km",
            "total_count": "Total",
        }
    )

    return sorted_counts


@table_method
def permitted_by_county():
    sorted_counts = facility_counts_by_county()
    # make an 'all other' category for counties with less than 10 facilities
    other = sorted_counts[sorted_counts["permitted_count"] < 10]
    sorted_counts = sorted_counts[sorted_counts["total_count"] >= 10]
    sorted_counts = sorted_counts.append(
        {
            "county": "All Other",
            "permitted_count": other["permitted_count"].sum(),
            "unpermitted_count": other["unpermitted_count"].sum(),
            "total_count": other["total_count"].sum(),
        },
        ignore_index=True,
    )
    total_row = pd.Series(
        {
            "county": "Total",
            "permitted_count": sorted_counts["permitted_count"].sum(),
            "unpermitted_count": sorted_counts["unpermitted_count"].sum(),
            "total_count": sorted_counts["total_count"].sum(),
        }
    )
    sorted_counts = sorted_counts.append(total_row, ignore_index=True)
    # rename columns
    sorted_counts = sorted_counts.rename(
        columns={
            "county": "County",
            "permitted_count": "Permit <1km",
            "unpermitted_count": "No Permit <1km",
            "total_count": "Total",
        }
    )
    return sorted_counts


@plot_method
def map_facility_counts_by_county():
    counts = facility_counts_by_county()
    county_gdf = gpd.GeoDataFrame(
        County.select(County.name, County.geometry).dicts(),
        crs="EPSG:4326",
    )
    county_gdf = county_gdf.merge(
        counts,
        how="left",
        left_on="name",
        right_on="county",
    ).fillna(0)
    county_gdf.plot(
        column="permitted_count",
        cmap="viridis",
        legend=True,
        legend_kwds={"label": "Permitted Facilities"},
        edgecolor="white",
        linewidth=0.3,
    )
    plt.title("Permitted Facilities by County")
    plt.axis("off")


@plot_method
def map_facility_counts_by_county_unpermitted():
    counts = facility_counts_by_county()
    county_gdf = gpd.GeoDataFrame(
        County.select(County.name, County.geometry).dicts(),
        crs="EPSG:4326",
    )
    county_gdf = county_gdf.merge(
        counts,
        how="left",
        left_on="name",
        right_on="county",
    ).fillna(0)
    county_gdf.plot(
        column="unpermitted_count",
        cmap="viridis",
        legend=True,
        legend_kwds={"label": "Unpermitted Facilities"},
        edgecolor="white",
        linewidth=0.3,
    )
    plt.title("Unpermitted Facilities by County")
    plt.axis("off")


@plot_method
def map_facility_locations():
    county_gdf = gpd.GeoDataFrame(
        County.select(County.name, County.geometry).dicts(),
        crs="EPSG:4326",
    )
    facility_gdf = gpd.GeoDataFrame(
        Facility.select(
            Facility.id,
            Facility.longitude,
            Facility.latitude,
            FacilityPermittedLocation.id.is_null(False).alias("permitted"),
        )
        .join(
            FacilityPermittedLocation,
            pw.JOIN.LEFT_OUTER,
        )
        .dicts()
    )
    facility_gdf["geometry"] = gpd.points_from_xy(
        facility_gdf["longitude"],
        facility_gdf["latitude"],
    )
    facility_gdf.crs = "EPSG:4326"

    facility_gdf["plot_permit"] = facility_gdf["permitted"].apply(
        lambda x: "Permit within 1km" if x else "No permit detected"
    )

    base = facility_gdf.plot(
        column="plot_permit",
        legend=True,
        markersize=0.07,
        marker="o",
        alpha=0.3,
        categorical=True,
        cmap="Set2",
        vmin=0,
        vmax=8,
        categories=["Permit within 1km", "No permit detected"],
        legend_kwds={
            "fontsize": 6,
            "markerscale": 0.5,
            "frameon": False,
            "borderpad": 0.1,
            # reduce space between markers and labels
            "handletextpad": 0.1,
        },
    )
    base = county_gdf.plot(
        color="none",
        edgecolor="black",
        linewidth=0.2,
        ax=base,
    )

    plt.title("Facility Detections")
    base.set_axis_off()


@plot_method
def number_of_facilities_over_time():
    facilities = pd.DataFrame(
        {
            "year": range(1993, 2024),
        }
    )
    facilities["count"] = facilities["year"].apply(
        lambda x: Facility.select()
        .join(ConstructionAnnotation)
        .where(
            (
                (ConstructionAnnotation.construction_upper_bound <= x)
                | ConstructionAnnotation.construction_lower_bound.is_null(True)
            )
            & (
                (ConstructionAnnotation.destruction_lower_bound >= x)
                | ConstructionAnnotation.destruction_lower_bound.is_null(True)
            )
        )
        .count()
    )
    # add more rows to the dataframe with the same years but different counts
    facilities = facilities.append(
        pd.DataFrame(
            {
                "year": facilities["year"],
                "count": facilities["year"].apply(
                    lambda x: Facility.select()
                    .join(ConstructionAnnotation)
                    .where(
                        (
                            (ConstructionAnnotation.construction_lower_bound <= x)
                            | ConstructionAnnotation.construction_lower_bound.is_null(
                                True
                            )
                        )
                        & (
                            (ConstructionAnnotation.destruction_upper_bound >= x)
                            | ConstructionAnnotation.destruction_upper_bound.is_null(
                                True
                            )
                        )
                    )
                    .count()
                ),
            }
        )
    )
    facilities = facilities.sort_values("year")
    facilities = facilities.reset_index(drop=True)

    sns.lineplot(
        data=facilities,
        x="year",
        y="count",
        label="Number of Facilities",
        estimator="mean",
        ls="--",
        linewidth=0.5,
        legend=False,
    )

    count = ConstructionAnnotation.select().count()
    plt.axhline(
        y=count,
        color="black",
        ls="--",
        linewidth=0.5,
        alpha=0.3,
    )
    # just abive the line
    plt.text(
        1992,
        count + 2,
        "Total Number of Facilities Detected in 2017",
        fontsize=6,
        alpha=0.5,
    )

    plt.ylim(1960, 2200)

    plt.title("Facility Counts Over Time, 2017 NAIP Cohort")
    plt.xlabel("Year")
    plt.ylabel("Number of Facilities")


@plot_method
def number_of_constructions_per_year():
    facilities = constructions_and_destructions_by_year()
    # only started and completed constructions
    facilities = facilities[
        facilities["event"].isin(
            [
                "started_constructions",
                "completed_constructions",
            ]
        )
    ]
    sns.lineplot(
        data=facilities,
        x="year",
        y="count",
        style="event",
        style_order=[
            "completed_constructions",
            "started_constructions",
        ],
        alpha=0.8,
    )
    plt.title("Construction Events Observed Per Year, 2017 NAIP Cohort")
    plt.xlabel("Year")
    plt.ylabel("Number of Events")
    plt.legend(
        labels=["Started", "Completed"],
        title=None,
        fontsize=6,
        frameon=False,
        handles=[
            plt.Line2D(
                [],
                [],
                linestyle="--",
            ),
            plt.Line2D(
                [],
                [],
                linestyle="-",
            ),
        ],
    )


@plot_method
def number_of_destructions_per_year():
    facilities = constructions_and_destructions_by_year()
    # only started and completed constructions
    facilities = facilities[
        facilities["event"].isin(
            [
                "started_destructions",
                "completed_destructions",
            ]
        )
    ]
    sns.lineplot(
        data=facilities,
        x="year",
        y="count",
        style="event",
        style_order=[
            "completed_destructions",
            "started_destructions",
        ],
        alpha=0.8,
    )
    plt.title("Destruction Event Observed Per Year, 2017 NAIP Cohort")
    plt.xlabel("Year")
    plt.ylabel("Number of Events")
    plt.legend(
        labels=["Started", "Completed"],
        title=None,
        fontsize=6,
        frameon=False,
        handles=[
            plt.Line2D(
                [],
                [],
                linestyle="--",
            ),
            plt.Line2D(
                [],
                [],
                linestyle="-",
            ),
        ],
    )


@plot_method
def map_example_permitted_facilities(plot_permits=False):
    permitted_facilities = list(
        Facility.select(
            Facility.id,
        )
        .join(
            FacilityPermittedLocation,
        )
        .distinct()
    )
    random.seed(1)
    permitted_facilities = random.sample(permitted_facilities, 4)
    gdfs = [facility.to_gdf() for facility in permitted_facilities]
    permit_locations = [
        list(
            FacilityPermittedLocation.select(
                PermittedLocation.longitude,
                PermittedLocation.latitude,
            )
            .join(
                PermittedLocation,
            )
            .where(
                FacilityPermittedLocation.facility == pf,
            )
            .dicts()
        )
        for pf in permitted_facilities
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i, (ax, gdf, permits) in enumerate(zip(axes.flatten(), gdfs, permit_locations)):
        gdf.plot(
            color=sns.color_palette("Set2")[3],
            alpha=0.6,
            edgecolor="black",
            linewidth=0.8,
            ax=ax,
        )
        # add permit detected locations
        if plot_permits:
            permit_gdf = gpd.GeoDataFrame(
                [
                    {
                        "geometry": shp.geometry.Point(
                            permit["longitude"],
                            permit["latitude"],
                        )
                    }
                    for permit in permits
                ]
            )
            permit_gdf.crs = "EPSG:4326"
            permit_gdf.plot(
                color="black",
                alpha=1,
                markersize=40,
                ax=ax,
            )
            permit_gdf.plot(
                color=sns.color_palette("Set2")[1],
                alpha=1,
                markersize=25,
                ax=ax,
            )

        # get longest axis
        longest_axis = max(
            ax.get_xlim()[1] - ax.get_xlim()[0],
            ax.get_ylim()[1] - ax.get_ylim()[0],
        )
        # extend each axis to be equal to longest axis
        ax.set_xlim(
            ax.get_xlim()[0]
            - (longest_axis - (ax.get_xlim()[1] - ax.get_xlim()[0])) / 2,
            ax.get_xlim()[1]
            + (longest_axis - (ax.get_xlim()[1] - ax.get_xlim()[0])) / 2,
        )
        ax.set_ylim(
            ax.get_ylim()[0]
            - (longest_axis - (ax.get_ylim()[1] - ax.get_ylim()[0])) / 2,
            ax.get_ylim()[1]
            + (longest_axis - (ax.get_ylim()[1] - ax.get_ylim()[0])) / 2,
        )
        naip.add_basemap(ax)
        ax.axis("off")


@plot_method
def map_example_unpermitted_facilities():
    unpermitted_facilities = list(
        Facility.select(
            Facility.id,
        )
        .join(
            FacilityPermittedLocation,
            pw.JOIN.LEFT_OUTER,
        )
        .where(
            FacilityPermittedLocation.id.is_null(True),
        )
        .distinct()
    )
    random.seed(1)
    unpermitted_facilities = random.sample(unpermitted_facilities, 4)
    gdfs = [facility.to_gdf() for facility in unpermitted_facilities]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        gdfs[i].plot(
            color=sns.color_palette("Set2")[3],
            alpha=0.6,
            edgecolor="black",
            linewidth=0.8,
            ax=ax,
        )
        # get longest axis
        longest_axis = max(
            ax.get_xlim()[1] - ax.get_xlim()[0],
            ax.get_ylim()[1] - ax.get_ylim()[0],
        )
        # extend each axis to be equal to longest axis
        ax.set_xlim(
            ax.get_xlim()[0]
            - (longest_axis - (ax.get_xlim()[1] - ax.get_xlim()[0])) / 2,
            ax.get_xlim()[1]
            + (longest_axis - (ax.get_xlim()[1] - ax.get_xlim()[0])) / 2,
        )
        ax.set_ylim(
            ax.get_ylim()[0]
            - (longest_axis - (ax.get_ylim()[1] - ax.get_ylim()[0])) / 2,
            ax.get_ylim()[1]
            + (longest_axis - (ax.get_ylim()[1] - ax.get_ylim()[0])) / 2,
        )

        naip.add_basemap(ax)
        ax.axis("off")


def map_clustering_example():
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    facility = (
        Facility.select()
        .join(Building)
        .group_by(Facility)
        .having((pw.fn.COUNT(Building.id) > 10) & (pw.fn.COUNT(Building.id) < 20))
        .get()
    )
    gdf = facility.to_gdf()
    for a in axes.flatten():
        gdf.plot(ax=a, linewidth=0.8)

    # get building relationships for the facility
    OtherBuilding = Building.alias()
    building_relationships = list(
        BuildingRelationship.select(
            Building.latitude.alias("latitude"),
            Building.longitude.alias("longitude"),
            OtherBuilding.latitude.alias("other_latitude"),
            OtherBuilding.longitude.alias("other_longitude"),
            OtherBuilding.id.alias("other_building"),
            BuildingRelationship.weight,
            BuildingRelationship.reason,
        )
        .join(Building, on=BuildingRelationship.building == Building.id)
        .join(OtherBuilding, on=BuildingRelationship.other_building == OtherBuilding.id)
        .where(Building.facility == facility)
        .dicts()
    )
    other_facilities = list(
        Facility.select()
        .join(Building)
        .where(Building.id.in_([br["other_building"] for br in building_relationships]))
        .distinct()
    )
    # plot other facilities in other colors
    for color, other_facility in zip(sns.color_palette("Set2")[1:], other_facilities):
        gdf = other_facility.to_gdf()
        for a in axes.flatten():
            gdf.plot(ax=a, color=color, linewidth=0.8)

    # turn off axis ticks
    for a in axes.flatten():
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])

    reason_to_axis = {
        "matching parcel": axes[0, 0],
        "distance": axes[0, 1],
        "parcel name tf-idf": axes[1, 0],
        "parcel name fuzzy": axes[1, 1],
    }
    reason_thresholds = {
        "matching parcel": -1,
        "distance": 400,
        "parcel name tf-idf": 700,
        "parcel name fuzzy": 600,
    }
    for relationship in building_relationships:
        reason_to_axis[relationship["reason"]].plot(
            [relationship["longitude"], relationship["other_longitude"]],
            [relationship["latitude"], relationship["other_latitude"]],
            color="black"
            if (relationship["weight"] is None)
            or (relationship["weight"] > reason_thresholds[relationship["reason"]])
            else "red",
            alpha=(relationship["weight"] or 1000) / 10000,
        )
    owner_names = list(
        Parcel.select(Parcel.owner)
        .join(Building)
        .join(Facility)
        .where(Facility.id.in_([facility.id] + [of.id for of in other_facilities]))
        .distinct()
        .dicts()
    )
    # add titles to each axis
    reason_to_axis["parcel name tf-idf"].set_title("Parcel Owner Name TF-IDF")
    reason_to_axis["parcel name fuzzy"].set_title("Parcel Owner Name Fuzzy Matching")
    reason_to_axis["matching parcel"].set_title("Matching Parcel")
    reason_to_axis["distance"].set_title("Distance")


@table_method
def recall():
    _Image = Image.alias()
    images_with_buildings = _Image.select(_Image.id).join(Building).distinct()

    unlabeled_image = Image.label_status == "unlabeled"
    post_hoc_image = (Image.label_status == "post hoc permit") | (
        Image.label_status == "adjacent"
    )
    unsampled_image = unlabeled_image | post_hoc_image
    sampled_image = Image.label_status == "active learner"

    positive_image = images_with_buildings.c.id.is_null(False)
    sampled_positive_image = sampled_image & positive_image

    query = (
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
        .where(Image.label_status != "removed")
        .group_by(Image.bucket)
        .order_by(Image.bucket)
    )
    df = pd.DataFrame(list(query.dicts()))

    # add total row
    total_row = pd.Series(
        {
            "bucket": "Total",
            "unlabeled": df["unlabeled"].sum(),
            "unsampled": df["unsampled"].sum(),
            "sampled": df["sampled"].sum(),
            "post_hoc_labeled": df["post_hoc_labeled"].sum(),
            "sampled_positive": df["sampled_positive"].sum(),
            "positive": df["positive"].sum(),
            "total": df["total"].sum(),
        }
    )
    df = df.append(total_row, ignore_index=True)

    df["sampled_prevalence_lower"] = df.apply(
        lambda x: proportion_confint(
            x["sampled_positive"], x["sampled"], method="beta"
        )[0],
        axis=1,
    )
    df["sampled_prevalence"] = df["sampled_positive"] / df["sampled"]
    df["sampled_prevalence_upper"] = df.apply(
        lambda x: proportion_confint(
            x["sampled_positive"], x["sampled"], method="beta"
        )[1],
        axis=1,
    )

    df["population_estimate_lower"] = (
        df["sampled_prevalence_lower"] * df["unlabeled"] + df["sampled_positive"]
    )
    df["population_estimate"] = (
        df["sampled_prevalence"] * df["unlabeled"] + df["sampled_positive"]
    )
    df["population_estimate_upper"] = (
        df["sampled_prevalence_upper"] * df["unlabeled"] + df["sampled_positive"]
    )

    for estimate in [
        "population_estimate_lower",
        "population_estimate",
        "population_estimate_upper",
    ]:
        df.loc[df["bucket"] == "Total", estimate] = 0
        df.loc[df["bucket"] == "Total", estimate] = sum(df[estimate])

    # bootstrap sum
    unsampled_buckets = df[(df["unsampled"] > 0) & (df["bucket"] != "Total")]
    sampled_buckets = df[(df["unsampled"] == 0) & (df["bucket"] != "Total")]
    other_totals = sum(sampled_buckets["positive"])
    trials = [
        np.random.binomial(
            row["sampled"],
            row["sampled_prevalence"],
            size=BOOTSTRAP_ITERS,
        )
        for _, row in unsampled_buckets.iterrows()
    ]
    population_estimates = sum(
        [
            row["unlabeled"] * trial / row["sampled"] + row["positive"]
            for trial, (_, row) in zip(trials, unsampled_buckets.iterrows())
        ]
    )
    population_estimates += other_totals
    df.loc[df["bucket"] == "Total", "population_estimate_lower"] = np.percentile(
        population_estimates,
        2.5,
    )
    df.loc[df["bucket"] == "Total", "population_estimate_upper"] = np.percentile(
        population_estimates,
        97.5,
    )

    df["recall_lower"] = df["positive"] / df["population_estimate_upper"]
    df["recall"] = df["positive"] / df["population_estimate"]
    df["recall_upper"] = df["positive"] / df["population_estimate_lower"]

    # round floats to 3 decimal places and population estimates to integers
    for col in [
        "sampled_prevalence_lower",
        "sampled_prevalence",
        "sampled_prevalence_upper",
        "population_estimate_lower",
        "population_estimate",
        "population_estimate_upper",
        "recall_lower",
        "recall",
        "recall_upper",
    ]:
        df[col] = df[col].apply(lambda x: round(x, 3))
    for col in [
        "unlabeled",
        "unsampled",
        "sampled",
        "post_hoc_labeled",
        "sampled_positive",
        "positive",
        "total",
    ]:
        df[col] = df[col].astype(int)

    # consolidate upper, lower, and point estimates into one col, format (upper, point, lower)
    # 3 decimal places
    df["Sampled Prevalence"] = df.apply(
        lambda x: f"({x['sampled_prevalence_upper']:0.3}, {x['sampled_prevalence']:0.3}, {x['sampled_prevalence_lower']:0.3})",
        axis=1,
    )
    df["Population Estimate"] = df.apply(
        lambda x: f"({x['population_estimate_upper']}, {x['population_estimate']}, {x['population_estimate_lower']})",
        axis=1,
    )
    df["Recall"] = df.apply(
        lambda x: f"({x['recall_upper']:0.3}, {x['recall']:0.3}, {x['recall_lower']:0.3})",
        axis=1,
    )
    # drop unlabeled, unsampled, sampled, post_hoc_labeled, sampled_positive, positive
    df = df.drop(
        columns=[
            "unlabeled",
            "unsampled",
            "sampled",
            "post_hoc_labeled",
            "sampled_positive",
            "positive",
            "sampled_prevalence_lower",
            "sampled_prevalence",
            "sampled_prevalence_upper",
            "population_estimate_lower",
            "population_estimate",
            "population_estimate_upper",
            "recall_lower",
            "recall",
            "recall_upper",
        ]
    )

    df = df.rename(
        columns={
            "bucket": "Bucket",
            "unlabeled": "Unlabeled",
            "unsampled": "Unsampled",
            "sampled": "Sampled",
            "post_hoc_labeled": "Post-Hoc Labeled",
            "sampled_positive": "Sampled Positive",
            "positive": "Positive",
            "total": "Total",
            "sampled_prevalence_lower": "Sampled Prevalence Lower",
            "sampled_prevalence": "Sampled Prevalence",
            "sampled_prevalence_upper": "Sampled Prevalence Upper",
            "population_estimate_lower": "Population Estimate Lower",
            "population_estimate": "Population Estimate",
            "population_estimate_upper": "Population Estimate Upper",
            "recall_lower": "Recall Lower",
            "recall": "Recall",
            "recall_upper": "Recall Upper",
        }
    )
    return df


@table_method
def labeling():
    _Image = Image.alias()
    images_with_buildings = _Image.select(_Image.id).join(Building).distinct()

    unlabeled_image = Image.label_status == "unlabeled"
    post_hoc_image = (Image.label_status == "post hoc permit") | (
        Image.label_status == "adjacent"
    )
    unsampled_image = unlabeled_image | post_hoc_image
    sampled_image = Image.label_status == "active learner"

    positive_image = images_with_buildings.c.id.is_null(False)
    sampled_positive_image = sampled_image & positive_image

    query = (
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
        .where(Image.label_status != "removed")
        .group_by(Image.bucket)
        .order_by(Image.bucket)
    )
    df = pd.DataFrame(list(query.dicts()))

    # add total row
    total_row = pd.Series(
        {
            "bucket": "Total",
            "unlabeled": df["unlabeled"].sum(),
            "unsampled": df["unsampled"].sum(),
            "sampled": df["sampled"].sum(),
            "post_hoc_labeled": df["post_hoc_labeled"].sum(),
            "sampled_positive": df["sampled_positive"].sum(),
            "positive": df["positive"].sum(),
            "total": df["total"].sum(),
        }
    )
    df = df.append(total_row, ignore_index=True)

    df["sampled_prevalence_lower"] = df.apply(
        lambda x: proportion_confint(
            x["sampled_positive"], x["sampled"], method="beta"
        )[0],
        axis=1,
    )
    df["sampled_prevalence"] = df["sampled_positive"] / df["sampled"]
    df["sampled_prevalence_upper"] = df.apply(
        lambda x: proportion_confint(
            x["sampled_positive"], x["sampled"], method="beta"
        )[1],
        axis=1,
    )

    df["population_estimate_lower"] = (
        df["sampled_prevalence_lower"] * df["unlabeled"] + df["positive"]
    )
    df["population_estimate"] = (
        df["sampled_prevalence"] * df["unlabeled"] + df["positive"]
    )
    df["population_estimate_upper"] = (
        df["sampled_prevalence_upper"] * df["unlabeled"] + df["positive"]
    )

    for estimate in [
        "population_estimate_lower",
        "population_estimate",
        "population_estimate_upper",
    ]:
        df.loc[df["bucket"] == "Total", estimate] = 0
        df.loc[df["bucket"] == "Total", estimate] = sum(df[estimate])

    # bootstrap sum
    unsampled_buckets = df[(df["unsampled"] > 0) & (df["bucket"] != "Total")]
    sampled_buckets = df[(df["unsampled"] == 0) & (df["bucket"] != "Total")]
    other_totals = sum(sampled_buckets["positive"])
    trials = [
        np.random.binomial(
            row["sampled"],
            row["sampled_prevalence"],
            size=BOOTSTRAP_ITERS,
        )
        for _, row in unsampled_buckets.iterrows()
    ]
    population_estimates = sum(
        [
            row["unlabeled"] * trial / row["sampled"] + row["positive"]
            for trial, (_, row) in zip(trials, unsampled_buckets.iterrows())
        ]
    )
    population_estimates += other_totals
    df.loc[df["bucket"] == "Total", "population_estimate_lower"] = np.percentile(
        population_estimates,
        2.5,
    )
    df.loc[df["bucket"] == "Total", "population_estimate_upper"] = np.percentile(
        population_estimates,
        97.5,
    )

    df["recall_lower"] = df["positive"] / df["population_estimate_upper"]
    df["recall"] = df["positive"] / df["population_estimate"]
    df["recall_upper"] = df["positive"] / df["population_estimate_lower"]

    for col in [
        "sampled_prevalence_lower",
        "sampled_prevalence",
        "sampled_prevalence_upper",
        "population_estimate_lower",
        "population_estimate",
        "population_estimate_upper",
        "recall_lower",
        "recall",
        "recall_upper",
    ]:
        df[col] = df[col].apply(lambda x: round(x, 3))
    for col in [
        "unlabeled",
        "unsampled",
        "sampled",
        "post_hoc_labeled",
        "sampled_positive",
        "positive",
        "total",
    ]:
        df[col] = df[col].astype(int)

    # drop all float cols
    df = df.drop(
        columns=[
            "sampled_prevalence_lower",
            "sampled_prevalence",
            "sampled_prevalence_upper",
            "population_estimate_lower",
            "population_estimate",
            "population_estimate_upper",
            "recall_lower",
            "recall",
            "recall_upper",
        ]
    )

    df = df.rename(
        columns={
            "bucket": "Bucket",
            "unlabeled": "Unlabeled",
            "unsampled": "Unsampled",
            "sampled": "Sampled",
            "post_hoc_labeled": "Post-Hoc Labeled",
            "sampled_positive": "Sampled Positive",
            "positive": "Positive",
            "total": "Total",
            "sampled_prevalence_lower": "Sampled Prevalence Lower",
            "sampled_prevalence": "Sampled Prevalence",
            "sampled_prevalence_upper": "Sampled Prevalence Upper",
            "population_estimate_lower": "Population Estimate Lower",
            "population_estimate": "Population Estimate",
            "population_estimate_upper": "Population Estimate Upper",
            "recall_lower": "Recall Lower",
            "recall": "Recall",
            "recall_upper": "Recall Upper",
        }
    )
    return df


@table_method
def tf_idf_examples():
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
            .limit(1)
            .dicts()
        )
        rows.extend(examples)
    rows = sorted(rows, key=lambda x: x["weight"])
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "weight": "TF-IDF Weight",
            "owner_1": "Owner 1",
            "owner_2": "Owner 2",
        }
    )
    return df


@table_method
def fuzzy_examples():
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
            .limit(1)
            .distinct()
            .dicts()
        )
        rows.extend(examples)
    rows = sorted(rows, key=lambda x: x["weight"])
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "weight": "Fuzzy Weight",
            "owner_1": "Owner 1",
            "owner_2": "Owner 2",
        }
    )
    return df


@table_method
def parcel_name_overrides():
    OtherBuilding = Building.alias()
    OtherParcel = Parcel.alias()
    overrides = (
        BuildingRelationship.select(
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
        .where((BuildingRelationship.reason == "parcel name annotation"))
        .distinct()
        .dicts()
    )
    pairs = {tuple(sorted((o["owner_1"], o["owner_2"]))) for o in overrides}
    df = pd.DataFrame(pairs, columns=["Owner 1", "Owner 2"])
    df = df.rename(
        columns={
            "owner_1": "Owner 1",
            "owner_2": "Owner 2",
        }
    )
    return df


@cache.memoize()
def facility_set(**kwargs):
    return [frozenset(f) for f in BuildingRelationship.facilities(**kwargs)]


# @plot_method
def facility_eps_relationship():
    num_facilities = []
    for eps in range(0, 1000, 50):
        facilities = facility_set(distance=eps)
        num_facilities.append(len(facilities))
    sns.lineplot(x=range(0, 1000, 50), y=num_facilities)
    plt.title("Number of Facilities by Maximum Distance Parameter")
    plt.xlabel("Distance Parameter (m)")
    plt.ylabel("Number of Facilities")


# @plot_method
def facility_fuzzy_relationship():
    num_facilities = []
    for fuzzy in range(0, 1000, 50):
        facilities = facility_set(fuzzy=fuzzy)
        num_facilities.append(len(facilities))
    sns.lineplot(x=[n / 1000 for n in range(0, 1000, 50)], y=num_facilities)
    plt.title("Number of Facilities by Fuzzy Matching Parameter")
    plt.xlabel("Fuzzy Matching Parameter")
    plt.ylabel("Number of Facilities")


# @plot_method
def facility_tfidf_relationship():
    num_facilities = []
    for tfidf in range(0, 1000, 50):
        facilities = facility_set(tfidf=tfidf)
        num_facilities.append(len(facilities))
    sns.lineplot(x=[n / 1000 for n in range(0, 1000, 50)], y=num_facilities)
    plt.title("Number of Facilities by TF-IDF Matching Parameter")
    plt.xlabel("TF-IDF Matching Parameter")
    plt.ylabel("Number of Facilities")


if __name__ == "__main__":
    written = set()
    pbar = tqdm(total=len(plot_methods) + len(table_methods))
    for pm in plot_methods[::-1]:
        pm()
        plt.savefig("paper/figures/eps/{}.eps".format(pm.__name__), bbox_inches="tight")
        plt.savefig("paper/figures/png/{}.png".format(pm.__name__), bbox_inches="tight")
        written.add("paper/figures/eps/{}.eps".format(pm.__name__))
        written.add("paper/figures/png/{}.png".format(pm.__name__))
        plt.close()
        pbar.update(1)
    for tm in table_methods[::-1]:
        df = tm()
        text = df.to_latex(index=False)
        with open("paper/tables/{}.tex".format(tm.__name__), "w") as f:
            f.write(text)
        pbar.update(1)
        written.add("paper/tables/{}.tex".format(tm.__name__))
        # pandoc to pdf
        os.system(
            "pandoc paper/tables/{}.tex -o paper/tables/pdf/{}.pdf".format(
                tm.__name__, tm.__name__
            )
        )
        written.add("paper/tables/pdf/{}.pdf".format(tm.__name__))
        df.to_csv("paper/tables/csv/{}.csv".format(tm.__name__), index=False)
    pbar.close()
    for file in itertools.chain(
        glob.glob("paper/tables/*.tex"),
        glob.glob("paper/figures/png/*.png"),
        glob.glob("paper/figures/eps/*.eps"),
        glob.glob("paper/figures/eps/*.pdf"),
    ):
        if file not in written:
            os.remove(file)
