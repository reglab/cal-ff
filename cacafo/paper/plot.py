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
import pandas as pd
import rich_click as click
import rl.utils.io
import seaborn as sns
import shapely as shp
import sqlalchemy as sa
from matplotlib_scalebar.scalebar import ScaleBar
from tqdm import tqdm

import cacafo.db.models as m
import cacafo.naip
import cacafo.query
from cacafo.cluster.buildings import building_clusters
from cacafo.db.session import new_session

sns.set_palette("Set2")
sns.set_theme("paper", style="white", font="Times New Roman")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (3, 2.5)
plt.rcParams["text.usetex"] = False
plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}\usepackage{amsmath}"

BOOTSTRAP_ITERS = 5000

facility_cluster_cache = dc.Cache(
    str(rl.utils.io.get_data_path("facility_cluster_cache"))
)


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
            TableMethod.path / "tex" / f"{self.name}.tex",
            TableMethod.path / "csv" / f"{self.name}.csv",
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
    session = new_session()
    permitted_facilities_subq = cacafo.query.permitted_cafos().subquery()
    unpermitted_facilities_subq = cacafo.query.unpermitted_cafos().subquery()
    # get rows of county, number of permitted facilities, number of unpermitted facilities
    permitted_counts = session.execute(
        sa.select(
            m.County.name,
            sa.func.count(sa.distinct(permitted_facilities_subq.c.id)).label(
                "permitted_count"
            ),
            sa.func.count(sa.distinct(unpermitted_facilities_subq.c.id)).label(
                "unpermitted_count"
            ),
        )
        .select_from(m.County)
        .outerjoin(
            permitted_facilities_subq,
            permitted_facilities_subq.c.county_id == m.County.id,
        )
        .outerjoin(
            unpermitted_facilities_subq,
            unpermitted_facilities_subq.c.county_id == m.County.id,
        )
        .group_by(m.County.name)
    )
    counts = pd.DataFrame(
        permitted_counts, columns=["county", "permitted_count", "unpermitted_count"]
    )
    counts["total_count"] = counts["permitted_count"] + counts["unpermitted_count"]
    counts = counts.sort_values("total_count", ascending=False)
    return counts


@figure()
def map_example_permitted_facilities():
    session = new_session()
    permitted_facilities = (
        session.scalars(cacafo.query.permitted_cafos().order_by(m.Facility.id))
        .unique()
        .all()
    )
    random.seed(7)
    permitted_facilities = random.sample(permitted_facilities, 4)
    map_facilities(permitted_facilities)


@figure()
def map_example_unpermitted_facilities():
    session = new_session()
    unpermitted_facilities = (
        session.scalars(cacafo.query.unpermitted_cafos().order_by(m.Facility.id))
        .unique()
        .all()
    )
    random.seed(7)
    unpermitted_facilities = random.sample(unpermitted_facilities, 4)
    map_facilities(unpermitted_facilities)


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
    session = new_session()
    permitted_facilities = session.scalars(
        cacafo.query.permitted_cafos()
        .options(
            sa.orm.joinedload(m.Facility.best_permits),
            sa.orm.joinedload(m.Facility.all_animal_type_annotations),
            sa.orm.Load(m.Facility).raiseload("*"),
        )
        .order_by(m.Facility.id)
    ).unique()
    unpermitted_facilities = session.scalars(
        cacafo.query.unpermitted_cafos()
        .options(
            sa.orm.joinedload(m.Facility.best_permits),
            sa.orm.joinedload(m.Facility.all_animal_type_annotations),
            sa.orm.Load(m.Facility).raiseload("*"),
        )
        .order_by(m.Facility.id)
    ).unique()
    df = pd.DataFrame(
        [
            {
                "animal_type": facility.animal_type_str.title(),
                "facility_id": facility.id,
                "permitted": True,
            }
            for facility in permitted_facilities
        ]
        + [
            {
                "animal_type": facility.animal_type_str.title(),
                "facility_id": facility.id,
                "permitted": False,
            }
            for facility in unpermitted_facilities
        ]
    )
    counts = df.groupby(["animal_type", "permitted"]).size().unstack().fillna(0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)
    counts = counts.reset_index()
    # make permit and total integers
    counts = counts.rename(
        columns={
            "animal_type": "Animal Type",
            False: "No Permit <1km",
            True: "Permit <1km",
            "total": "Total",
        }
    )
    # add total line at the bottom
    total_row = counts.sum()
    total_row["Animal Type"] = "Total"
    counts = pd.concat(
        [counts, pd.DataFrame([total_row], columns=counts.columns)],
        ignore_index=True,
    )
    counts["Total"] = counts["Total"].astype(int)
    counts["Permit <1km"] = counts["Permit <1km"].astype(int)
    counts["No Permit <1km"] = counts["No Permit <1km"].astype(int)
    return counts


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
    other = sorted_counts[sorted_counts["total_count"] < 50]
    sorted_counts = sorted_counts[sorted_counts["total_count"] >= 50]
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
def map_facility_locations():
    session = new_session()
    counties = session.scalars(sa.select(m.County)).all()
    county_gdf = gpd.GeoDataFrame(
        [{"name": county.name, "geometry": county.shp_geometry} for county in counties],
        geometry="geometry",
        crs="EPSG:4326",
    )
    permitted_ids = {f.id for f in session.scalars(cacafo.query.permitted_cafos())}
    cafos = session.scalars(
        cacafo.query.cafos().options(sa.orm.Load(m.Facility).raiseload("*"))
    )
    facility_gdf = gpd.GeoDataFrame(
        [
            {
                "id": facility.id,
                "latitude": facility.shp_geometry.centroid.y,
                "longitude": facility.shp_geometry.centroid.x,
                "permitted": facility.id in permitted_ids,
                "geometry": facility.shp_geometry.centroid,
            }
            for facility in cafos
        ]
    )
    facility_gdf.crs = "EPSG:4326"

    facility_gdf["plot_permit"] = facility_gdf["permitted"].apply(
        lambda x: "Permit <1km" if x else "No Permit <1km"
    )

    base = facility_gdf.plot(
        column="plot_permit",
        legend=True,
        markersize=0.07,
        marker="o",
        alpha=0.2,
        categorical=True,
        cmap="Set2",
        vmin=0,
        vmax=8,
        categories=["Permit <1km", "No Permit <1km"],
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


def map_facilities(facilities, permit_locations=[]):
    gdfs = [facility.gdf for facility in facilities]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    if not permit_locations:
        permit_locations = [None] * len(facilities)
    for i, (ax, gdf, permits) in enumerate(zip(axes.flatten(), gdfs, permit_locations)):
        gdf.crs = "EPSG:4326"
        gdf = gdf.to_crs("EPSG:3311")
        gdf.plot(
            facecolor="none",
            edgecolor="black",
            linewidth=0,
            ax=ax,
        )
        gdf.plot(
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            ax=ax,
        )
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
        ax.axis("off")
    # maxis = max([ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes.flatten()])
    maxis = 500
    for ax in axes.flatten():
        midpoint = (
            (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2,
            (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2,
        )
        ax.set_xlim(midpoint[0] - maxis / 2, midpoint[0] + maxis / 2)
        ax.set_ylim(midpoint[1] - maxis / 2, midpoint[1] + maxis / 2)
        ax.add_artist(ScaleBar(1))
        cacafo.naip.add_basemap(ax)
    plt.tight_layout()


@table()
def labeling():
    from cacafo.stats.population import Survey

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
    df = df[~df["name"].str.contains("post hoc")]
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df


@table()
def tf_idf_examples():
    session = new_session()
    related_building = sa.orm.aliased(m.Building)
    other_parcel = sa.orm.aliased(m.Parcel)
    tfidf_relationships = (
        session.execute(
            sa.select(
                m.BuildingRelationship.weight.label("weight"),
                m.Parcel.owner.label("owner_1"),
                other_parcel.owner.label("owner_2"),
            )
            .select_from(m.BuildingRelationship)
            .join(m.Building, (m.Building.id == m.BuildingRelationship.building_id))
            .join(
                related_building,
                (related_building.id == m.BuildingRelationship.related_building_id),
            )
            .join(m.Parcel, (m.Parcel.id == m.Building.parcel_id))
            .join(other_parcel, (other_parcel.id == related_building.parcel_id))
            .where(m.BuildingRelationship.reason == "tfidf")
            .order_by(m.BuildingRelationship.weight.desc())
            .distinct()
        )
        .mappings()
        .all()
    )
    df = pd.DataFrame(tfidf_relationships)
    df = df.rename(
        columns={
            "weight": "TF-IDF Weight",
            "owner_1": "Owner 1",
            "owner_2": "Owner 2",
        }
    )
    df["Owner 1"] = df["Owner 1"].apply(lambda x: x.replace("&", r"\&"))
    df["Owner 2"] = df["Owner 2"].apply(lambda x: x.replace("&", r"\&"))
    ranges = zip(range(0, 1000, 50), range(50, 1000, 50))

    # filter out names longer than 30 characters
    df = df[df["Owner 1"].apply(len) < 30]
    df = df[df["Owner 2"].apply(len) < 30]
    # sample one example from each range
    rows = []
    for low, high in ranges:
        example = df[(df["TF-IDF Weight"] >= low) & (df["TF-IDF Weight"] < high)].iloc[
            0
        ]
        rows.append(example)
    df = pd.DataFrame(rows)
    df["TF-IDF Weight"] = df["TF-IDF Weight"].astype(int)
    return df


@table()
def fuzzy_examples():
    session = new_session()
    related_building = sa.orm.aliased(m.Building)
    other_parcel = sa.orm.aliased(m.Parcel)
    fuzzy_relationships = (
        session.execute(
            sa.select(
                m.BuildingRelationship.weight.label("weight"),
                m.Parcel.owner.label("owner_1"),
                other_parcel.owner.label("owner_2"),
            )
            .select_from(m.BuildingRelationship)
            .join(m.Building, (m.Building.id == m.BuildingRelationship.building_id))
            .join(
                related_building,
                (related_building.id == m.BuildingRelationship.related_building_id),
            )
            .join(m.Parcel, (m.Parcel.id == m.Building.parcel_id))
            .join(other_parcel, (other_parcel.id == related_building.parcel_id))
            .where(m.BuildingRelationship.reason == "fuzzy")
            .order_by(m.BuildingRelationship.weight.desc())
            .distinct()
        )
        .mappings()
        .all()
    )
    df = pd.DataFrame(fuzzy_relationships)
    df = df.rename(
        columns={
            "weight": "Fuzzy Weight",
            "owner_1": "Owner 1",
            "owner_2": "Owner 2",
        }
    )
    df["Owner 1"] = df["Owner 1"].apply(lambda x: x.replace("&", r"\&"))
    df["Owner 2"] = df["Owner 2"].apply(lambda x: x.replace("&", r"\&"))
    ranges = zip(range(0, 1000, 50), range(50, 1000, 50))

    df = df[df["Owner 1"].apply(len) < 30]
    df = df[df["Owner 2"].apply(len) < 30]
    # sample one example from each range
    rows = []
    for low, high in ranges:
        example = df[(df["Fuzzy Weight"] >= low) & (df["Fuzzy Weight"] < high)].iloc[0]
        rows.append(example)

    df = pd.DataFrame(rows)
    df["Fuzzy Weight"] = df["Fuzzy Weight"].astype(int)
    return df


@table()
def parcel_name_overrides():
    session = new_session()
    rows = session.scalars(
        sa.select(m.ParcelOwnerNameAnnotation).where(
            m.ParcelOwnerNameAnnotation.matched
        )
    ).all()
    rows = {tuple(sorted((row.owner_name, row.related_owner_name))) for row in rows}
    df = pd.DataFrame(
        [
            {
                "Owner 1": row[0],
                "Owner 2": row[1],
            }
            for row in rows
        ]
    )
    df["Owner 1"] = df["Owner 1"].apply(lambda x: x.replace("&", r"\&"))
    df["Owner 2"] = df["Owner 2"].apply(lambda x: x.replace("&", r"\&"))
    return df


@table()
def county_groups():
    session = new_session()
    county_groups = (
        session.execute(
            sa.select(
                m.CountyGroup.name.label("County Group"),
                m.County.name.label("County"),
            )
            .select_from(m.CountyGroup)
            .join(m.County, m.CountyGroup.counties)
            .order_by(m.CountyGroup.name, m.County.name)
        )
        .mappings()
        .all()
    )
    df = pd.DataFrame(county_groups)
    df = df.groupby("County Group")["County"].apply(list).reset_index()
    df["n_counties"] = df["County"].apply(len)
    df = df.sort_values("n_counties", ascending=True)
    df = df[df["n_counties"] > 1]
    df = df.drop(columns=["n_counties"])
    df["Counties"] = df["County"].apply(lambda x: ", ".join(x))
    df = df.drop(columns=["County"])
    df["County Group"] = [f"Group {i}" for i, _ in enumerate(df["County Group"], 1)]
    return df


@facility_cluster_cache.memoize()
def facility_set(**kwargs):
    return [frozenset(f) for f in building_clusters(**kwargs)]


@figure()
def facility_matching_parameters():
    plt.figure(figsize=(7, 2.5))
    ax = plt.subplot(1, 3, 1)
    num_facilities = []
    for eps in range(0, 1000, 50):
        facilities = facility_set(distance=eps)
        num_facilities.append(len(facilities))
    sns.lineplot(x=range(0, 1000, 50), y=num_facilities, ax=ax)
    ax.axvline(400, color="lightgray", linestyle="--")
    ax.axvline(200, color="lightgray", linestyle="--")
    ax.set_xlabel("Distance Parameter (m)")
    ax.set_ylabel("Number of Facilities")
    ax.yaxis.set_major_locator(plt.MultipleLocator(200))

    ylim = ax.get_ylim()

    ax = plt.subplot(1, 3, 2)
    num_facilities = []
    for fuzzy in range(0, 1000, 50):
        facilities = facility_set(fuzzy=fuzzy)
        num_facilities.append(len(facilities))
    sns.lineplot(x=[n / 1000 for n in range(0, 1000, 50)], y=num_facilities, ax=ax)
    ax.axvline(0.6, color="lightgray", linestyle="--")
    ax.set_xlabel("Fuzzy Matching Parameter")
    ax.set_ylabel("Number of Facilities")
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(plt.MultipleLocator(200))

    ax = plt.subplot(1, 3, 3)
    num_facilities = []
    for tfidf in range(0, 1000, 50):
        facilities = facility_set(tfidf=tfidf)
        num_facilities.append(len(facilities))
    sns.lineplot(x=[n / 1000 for n in range(0, 1000, 50)], y=num_facilities, ax=ax)
    ax.axvline(0.7, color="lightgray", linestyle="--")
    ax.set_xlabel("TF-IDF Matching Parameter")
    ax.set_ylabel("Number of Facilities")
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(plt.MultipleLocator(200))

    plt.tight_layout()


@figure()
def permit_sensitivity_analysis():
    from cacafo.cluster.permits import facility_parcel_then_distance_matches

    matching_distances = range(0, 1001, 50)

    data = []
    for distance in matching_distances:
        data.append(
            {
                "distance": distance,
                "n_clean_matches": sum(
                    len(v)
                    for v in facility_parcel_then_distance_matches(
                        distance=distance
                    ).values()
                ),
            }
        )

    df = pd.DataFrame(data)
    sns.lineplot(
        data=df,
        x="distance",
        y="n_clean_matches",
    )
    plt.axvline(200, color="lightgray", linestyle="--")
    plt.title("Number of Best Permit Matches by Matching Distance")
    plt.xlabel("Matching Distance (m)")
    plt.ylabel("Number of Best Permit Matches")
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


@click.command("plot")
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
def _cli(item, output_path):
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
