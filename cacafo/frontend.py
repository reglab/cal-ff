import urllib.parse

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

import models as m

tile_server_url = "http://localhost:5000/tiles/{z}/{x}/{y}"

# Sample data for building bounding boxes
buildings = m.Facility.select().first().to_gdf()
buildings["lat"] = buildings.centroid.y
buildings["lon"] = buildings.centroid.x

# Create Dash app
app = dash.Dash(__name__)

# Create map with building bounding boxes
fig = px.scatter_mapbox(
    buildings,
    lat="lat",
    lon="lon",
    text="id",
    title="Building Bounding Boxes with Basemap",
    mapbox_style="open-street-map",
    center=dict(lat=buildings.centroid.y.mean(), lon=buildings.centroid.x.mean()),
    zoom=15,
)

if True:
    fig.update_layout(
        mapbox=dict(
            style="white-bg",
            center=dict(
                lat=buildings.centroid.y.mean(), lon=buildings.centroid.x.mean()
            ),
            zoom=15,
            layers=[
                {
                    "sourcetype": "raster",
                    "source": [tile_server_url],
                }
            ],
        ),
    )

# Create layout for Dash app
app.layout = html.Div([dcc.Graph(id="map", figure=fig), html.Div(id="building-info")])


# Callback to display building information on click
@app.callback(Output("building-info", "children"), [Input("map", "clickData")])
def display_building_info(click_data):
    if click_data is None:
        return ""

    # Extract building information from click data
    building_name = click_data["points"][0]["text"]
    building_id = buildings[buildings["id"] == building_name]["id"].values[0]

    # Display building information in a panel
    return html.Div(
        [html.H3(f"Building: {building_name}"), html.P(f"ID: {building_id}")]
    )


@app.callback(Output("map", "figure"), [Input("url", "search")])
def update_map_center(query):
    global buildings
    if query:
        # Parse latitude and longitude from query parameters
        params = urllib.parse.parse_qs(query)
        lat = float(params.get("lat", 0))
        lon = float(params.get("lon", 0))
        # Update map center and zoom level
        fig.update_geos(center=dict(lat=lat, lon=lon))


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
