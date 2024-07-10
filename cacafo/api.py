import io
import json
import math

import flask
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import peewee as pw
import requests
import shapely as shp
from flask import jsonify, request

import models as m
import naip

matplotlib.use("Agg")

app = flask.Flask(__name__)


def _get_bounds(lat, lon, size):
    gdf = gpd.GeoDataFrame(geometry=[shp.geometry.Point(lon, lat)], crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3311")
    gdf["geometry"] = gdf.buffer(size * 1000)
    gdf = gdf.to_crs("EPSG:4326")
    bounds = gdf.bounds
    return bounds


# an GET endpoint that take a lat/lon center and size in km
# all CAFOs and buildings
@app.route("/image", methods=["GET"])
def get_image():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    size = request.args.get("size")

    bounds = _get_bounds(float(lat), float(lon), float(size))
    bounds = shp.geometry.box(*bounds.values[0])

    buildings = (
        m.Building.select()
        .where(pw.fn.ST_Within(m.Building.geometry, bounds.wkt))
        .dicts()
    )
    gdf = gpd.GeoDataFrame.from_records(buildings)

    # annotate patches with the building id
    ax = gdf.plot()
    # set gid for each building

    for i, patch in enumerate(ax.patches):
        patch.set_gid(gdf.iloc[i]["id"])
        patch.set_label(gdf.iloc[i]["id"])

    ax.set_xlim(bounds.bounds[0], bounds.bounds[2])
    ax.set_ylim(bounds.bounds[1], bounds.bounds[3])
    # plot a clear rectangle around the bounds
    x, y = bounds.exterior.xy
    ax.plot(x, y, color="black")
    naip.add_basemap(ax)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # set ax flush with the figure
    ax.margins(0)
    fig = ax.get_figure()
    # set figure margins to 0
    fig.tight_layout()
    f = io.BytesIO()
    plt.savefig(f, format="svg")
    plt.close()
    f.seek(0)
    return f.getvalue(), 200, {"Content-Type": "image/svg+xml"}


@app.route("/tiles/<int:z>/<int:x>/<int:y>", methods=["GET"])
def get_tiles(z, x, y):
    img = naip.get_tile(z, x, y)
    output = io.BytesIO()
    img.save(output, format="png")
    output.seek(0)
    return output.getvalue(), 200, {"Content-Type": "image/png"}


# an GET endpoint that take a lat/lon center and size in km


if __name__ == "__main__":
    app.run(debug=True)
