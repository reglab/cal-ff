import abc
import itertools
import os
import random
import textwrap as tw
from functools import cache
from glob import glob
from pathlib import Path

import diskcache as dc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peewee as pw
import rich_click as click
import rl.utils.io
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

import cacafo.db.models as m
import cacafo.naip
from cacafo.cluster.buildings import building_clusters
from cacafo.db.models import *

sns.set_palette("Set2")
sns.set_theme("paper", style="white", font="Times New Roman")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (3, 2.5)
plt.rcParams["text.usetex"] = False
plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}\usepackage{amsmath}"

BOOTSTRAP_ITERS = 5000

facility_cluster_cache = dc.Cache("facility_cluster_cache")


class PaperMethod(abc.ABC):
    instances = []

    @classmethod
    def _register(cls, pm: "PaperMethod"):
        cls.instances.append(pm)

    def __init__(self, method, cache=True):
        self.method = method
        self.cache = cache
        PaperMethod._register(self)
        self.__class__._register(self)

    @property
    def name(self):
        return self.method.__name__

    @abc.abstractmethod
    def __call__(self):
        pass

    @abc.abstractmethod
    def save(self, path):
        """Save the output of this method to the given path with the appropriate format"""
        pass

    @abc.abstractmethod
    def paths(self):
        """Return a list of paths that this method will save to"""
        pass

    @abc.abstractmethod
    def save_all(self, base_path):
        """Save all outputs to the given base path with the appropriate names given by paths()"""
        pass


class TableMethod(PaperMethod):
    instances = []
    path = Path("tables")

    def __init__(self, method, header, footer, cache=True):
        super().__init__(method, cache)
        self.header = header
        self.footer = footer

    def __call__(self):
        if self.cache and hasattr(self, "_result"):
            return self._result
        self._result = self.method()
        return self._result

    def df(self):
        return self()

    def tex(self):
        df = self.method()
        text = df.to_latex(index=False)
        if self.header:
            text = "\n".join((self.header, text.split("\\midrule")[1]))
        if self.footer:
            text = "\n".join((text.split("\\bottomrule")[0], self.footer))
        return text

    def csv(self):
        df = self.method()
        return df.to_csv(index=False)

    def paths(self):
        return [
            TableMethod.path / f"{self.name}.tex",
            TableMethod.path / f"{self.name}.csv",
        ]

    def save(self, path):
        path = str(path)
        if not path.endswith(".tex") and not path.endswith(".csv"):
            raise ValueError("Path must end with .tex or .csv")
        if path.endswith(".tex"):
            with open(path, "w") as f:
                f.write(self.tex())
        elif path.endswith(".csv"):
            with open(path, "w") as f:
                f.write(self.csv())

    def save_all(self, base_path):
        for path in self.paths():
            self.save(base_path / path)


class FigureMethod(PaperMethod):
    instances = []
    path = Path("figures")

    def __call__(self):
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.sca(ax)
        self.method()
        # fig.tight_layout()
        # self._fig = fig
        # self._ax = ax
        # return self._fig

    def save(self, path):
        path = str(path)
        self()
        plt.savefig(
            path,
            bbox_inches="tight",
        )
        plt.close()

    def paths(self):
        return [
            FigureMethod.path / "png" / f"{self.name}.png",
            FigureMethod.path / "eps" / f"{self.name}.eps",
        ]

    def save_all(self, base_path):
        for path in self.paths():
            self.save(base_path / path)


def figure(cache=True):
    def decorator(method):
        return FigureMethod(method, cache)

    return decorator


def table(header=None, footer=None, cache=True):
    def decorator(method):
        return TableMethod(method, header, footer, cache)

    return decorator


@cache
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


@cache
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


@table(
    header=tw.dedent(
        r"""
        \begin{tabular}{lrrr}
        \toprule
        & \multicolumn{2}{c}{Permit within 1km} &\\ Animal Type & Yes & No &  Total \\
        \midrule
    """.strip()
    ),
)
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
    sorted_counts["animal_type"] = sorted_counts["animal_type"].apply(
        lambda s: s.title()
    )
    sorted_counts = sorted_counts.rename(
        columns={
            "animal_type": "Animal Type",
            "permitted_count": "Permit <1km",
            "unpermitted_count": "No Permit <1km",
            "total_count": "Total",
        }
    )

    return sorted_counts


@table(
    header=tw.dedent(
        r"""
        \begin{tabular}{lrrr}
        \toprule
        & \multicolumn{2}{c}{Permit within 1km} &\\ County & Yes & No &  Total \\
        \midrule
    """.strip()
    ),
)
def permitted_by_county():
    sorted_counts = facility_counts_by_county()
    # make an 'all other' category for counties with less than 10 facilities
    other = sorted_counts[sorted_counts["permitted_count"] < 10]
    sorted_counts = sorted_counts[sorted_counts["total_count"] >= 10]
    sorted_counts = pd.concat(
        [
            sorted_counts,
            pd.DataFrame(
                [
                    {
                        "county": "All Other",
                        "permitted_count": other["permitted_count"].sum(),
                        "unpermitted_count": other["unpermitted_count"].sum(),
                        "total_count": other["total_count"].sum(),
                    },
                ]
            ),
        ]
    )
    total_row = pd.Series(
        {
            "county": "Total",
            "permitted_count": sorted_counts["permitted_count"].sum(),
            "unpermitted_count": sorted_counts["unpermitted_count"].sum(),
            "total_count": sorted_counts["total_count"].sum(),
        }
    )
    sorted_counts = pd.concat(
        [
            sorted_counts,
            pd.DataFrame([total_row]),
        ]
    )
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


@figure()
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
        alpha=0.5,
    )
    plt.title("Permitted Facilities by County")
    plt.axis("off")


@figure()
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


@figure()
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
        lambda x: "Permit within 1km" if x else "No permit within 1km"
    )

    base = facility_gdf.plot(
        column="plot_permit",
        legend=True,
        markersize=0.07,
        marker="o",
        alpha=0.1,
        categorical=True,
        cmap="Set2",
        vmin=0,
        vmax=8,
        categories=["Permit within 1km", "No permit within 1km"],
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


@figure()
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


@figure()
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


@figure()
def map_example_permitted_facilities(plot_permits=False):
    permitted_facilities = list(
        Facility.select()
        .join(
            FacilityPermittedLocation,
        )
        .distinct()
    )
    random.seed(5)
    permitted_facilities = random.sample(permitted_facilities, 4)
    permit_locations = []
    if plot_permits:
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
            for pf in facilities
        ]
    map_facilities(permitted_facilities, permit_locations)


@figure()
def map_example_unpermitted_facilities():
    unpermitted_facilities = list(
        Facility.select()
        .join(
            FacilityPermittedLocation,
            pw.JOIN.LEFT_OUTER,
        )
        .where(
            FacilityPermittedLocation.id.is_null(True),
        )
        .distinct()
    )
    random.seed(5)
    unpermitted_facilities = random.sample(unpermitted_facilities, 4)
    map_facilities(unpermitted_facilities)


def map_facilities(facilities, permit_locations=[]):
    gdfs = [facility.to_gdf() for facility in facilities]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    if not permit_locations:
        permit_locations = [None] * len(facilities)
    for i, (ax, gdf, permits) in enumerate(zip(axes.flatten(), gdfs, permit_locations)):
        gdf.crs = "EPSG:4326"
        gdf = gdf.to_crs("EPSG:3311")
        gdf.plot(
            facecolor="none",
            edgecolor="black",
            linewidth=4,
            ax=ax,
        )
        gdf.plot(
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            ax=ax,
        )
        ax.add_artist(ScaleBar(1))
        # add permit detected locations
        if permits:
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
        cacafo.naip.add_basemap(ax)
        ax.axis("off")


@table()
def labeling():
    from cacafo.stats.population import Stratum, Survey

    survey = Survey.from_db()
    df = survey.to_df()
    df = df.sort_values("total")
    df = df.sort_values("positive", ascending=False)
    df["bucket"] = df["name"].apply(
        lambda x: r"high confidence \& permit"
        if "completed" in x
        else "low confidence"
        if "1:" in x
        else "no detection"
    )
    df["name"] = df["name"].apply(lambda x: x.replace("0:", ""))
    df["name"] = df["name"].apply(lambda x: x.replace("1:", ""))
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df


@table()
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
    df["Owner 1"] = df["Owner 1"].apply(lambda x: x.replace("&", r"\&"))
    df["Owner 2"] = df["Owner 2"].apply(lambda x: x.replace("&", r"\&"))
    return df


@table()
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
    df["Owner 1"] = df["Owner 1"].apply(lambda x: x.replace("&", r"\&"))
    df["Owner 2"] = df["Owner 2"].apply(lambda x: x.replace("&", r"\&"))
    return df


@table()
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
    df["Owner 1"] = df["Owner 1"].apply(lambda x: x.replace("&", r"\&"))
    df["Owner 2"] = df["Owner 2"].apply(lambda x: x.replace("&", r"\&"))
    return df


@facility_cluster_cache.memoize()
def facility_set(**kwargs):
    return [frozenset(f) for f in building_clusters(**kwargs)]


@figure()
def facility_eps_relationship():
    num_facilities = []
    for eps in range(0, 1000, 50):
        facilities = facility_set(distance=eps)
        num_facilities.append(len(facilities))
    sns.lineplot(x=range(0, 1000, 50), y=num_facilities)
    plt.title("Number of Facilities by Maximum Distance Parameter")
    plt.xlabel("Distance Parameter (m)")
    plt.ylabel("Number of Facilities")


@figure()
def facility_fuzzy_relationship():
    num_facilities = []
    for fuzzy in range(0, 1000, 50):
        facilities = facility_set(fuzzy=fuzzy)
        num_facilities.append(len(facilities))
    sns.lineplot(x=[n / 1000 for n in range(0, 1000, 50)], y=num_facilities)
    plt.title("Number of Facilities by Fuzzy Matching Parameter")
    plt.xlabel("Fuzzy Matching Parameter")
    plt.ylabel("Number of Facilities")


@figure()
def facility_tfidf_relationship():
    num_facilities = []
    for tfidf in range(0, 1000, 50):
        facilities = facility_set(tfidf=tfidf)
        num_facilities.append(len(facilities))
    sns.lineplot(x=[n / 1000 for n in range(0, 1000, 50)], y=num_facilities)
    plt.title("Number of Facilities by TF-IDF Matching Parameter")
    plt.xlabel("TF-IDF Matching Parameter")
    plt.ylabel("Number of Facilities")


@figure()
def permit_sensitivity_analysis():
    from cacafo.cluster.permits import (
        _conjunction_dict_of_sets,
        _disjunction_dict_of_sets,
        _remove_duplicate_entries,
        _remove_empty_entries,
        facility_permit_distance_matches,
        facility_permit_parcel_matches,
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
            } | row
            data.append(parcel_or_distance_row)

            parcel_then_distance_matches = _disjunction_dict_of_sets(
                parcel_matches,
                _remove_empty_entries(_remove_duplicate_entries(matches)),
            )
            parcel_then_distance_row = {
                "matching_method": "parcel then distance",
                "n_clean_matches": len(parcel_then_distance_matches),
            } | row
            data.append(parcel_then_distance_row)

    df = pd.DataFrame(data)
    # only both and parcel then distance
    df = df[df["matching_method"] == "parcel then distance"]
    df = df[df["location_type"] == "both"]
    # make lineplot
    sns.lineplot(
        data=df,
        x="distance",
        y="n_clean_matches",
    )
    plt.title("Number of Clean Matches by Matching Distance")
    plt.xlabel("Matching Distance (m)")

    plt.ylabel("Number of Clean Matches")
    plt.tight_layout()
    return df


def generate(output_path, items=None):
    paper_items = PaperMethod.instances
    if items:
        paper_items = [p for p in paper_items if p.name in items]
    pbar = tqdm(total=len(paper_items))
    for item in paper_items:
        item.save_all(output_path)
        pbar.update(1)


@click.command("paper")
@click.option(
    "--item",
    "-i",
    default=None,
    type=click.Choice([p.name for p in PaperMethod.instances]),
    help="Generate a specific item",
    multiple=True,
)
@click.option(
    "--output-path",
    "-o",
    default=None,
    type=click.Path(),
    help="Output path; defaults to DATA_ROOT/paper/figures and DATA_ROOT/paper/tables",
)
def cmd_generate(item, output_path):
    if output_path is None:
        output_path = str(rl.utils.io.get_data_path() / "paper")

    if not item:
        for file in itertools.chain(
            glob(f"{output_path}/tables/*.tex"),
            glob(f"{output_path}/figures/png/*.png"),
            glob(f"{output_path}/figures/eps/*.eps"),
            glob(f"{output_path}/figures/eps/*.pdf"),
        ):
            os.remove(file)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "figures", "png"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "figures", "eps"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "tables", "tex"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "tables", "csv"), exist_ok=True)
    generate(output_path, item)
