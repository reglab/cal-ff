import os
from typing import List

import rich
import seaborn as sns
import shapely as shp
from tqdm import tqdm

import cacafo.naip as naip
from cacafo.geom import clean_building_geometry

PROJECT = "RegLab_Prod"


def dl_auth():
    import dtlpy as dl

    if dl.token_expired():
        dl.login()


def create_labeling_dataset(prefix, images):
    import dtlpy as dl

    image_names = [image.name for image in images]
    already_exists = set(
        [
            os.path.basename(image).split(".")[0]
            for image in naip.list_images_with_prefix(prefix)
        ]
    )
    to_upload = [image for image in image_names if image not in already_exists]
    if to_upload:
        naip.create_subset(prefix, to_upload)

    dl_auth()
    project = dl.projects.get(PROJECT)
    prefix = prefix.strip("/")
    drivers = project.drivers.list()
    driver = None
    for existing_driver in drivers[::-1]:
        if existing_driver.name == prefix:
            if existing_driver.path == prefix:
                driver = existing_driver
                break
            else:
                raise ValueError(
                    f"Driver with name {prefix} already exists but has different path {existing_driver.path}."
                )
    if not driver:
        driver = project.drivers.create(
            name=prefix,
            driver_type=dl.ExternalStorage.GCS,
            # for law-cafo, gotten through net request spy
            integration_id="41b5b0c8-bf3f-4054-ab44-edd7718768a8",
            integration_type=dl.IntegrationType.GCS,
            bucket_name="image-hub",
            path=prefix,
        )

    labels = ["Blank", "flag", "cafo"]
    colors = sns.color_palette("hls", len(labels))
    labels_dict = {}
    for i, label in enumerate(labels):
        color = colors[i]
        color = [int(c * 255) for c in color]
        labels_dict[label] = tuple(color)

    datasets = project.datasets.list()
    dataset = None
    for existing_dataset in datasets:
        if existing_dataset.name == prefix and existing_dataset.driver != driver.id:
            raise ValueError(
                f"Dataset with name {prefix} already exists but has different driver {dataset.driver.id}."
            )
        if existing_dataset.name == prefix and existing_dataset.driver == driver.id:
            dataset = existing_dataset
    if not dataset:
        dataset = project.datasets.create(
            driver=driver,
            dataset_name=prefix,
            labels=labels_dict,
        )
        dataset.sync()

    for image in tqdm(images):
        name = f"{image.name}.jpeg"
        item = dataset.items.get("/" + name)
        geometry = image.shp_geometry
        lat, lon = geometry.centroid.y, geometry.centroid.x
        item.metadata["user"] = {
            "latitude": lat,
            "longitude": lon,
            "gmaps_link": f"https://www.google.com/maps/place/{lat},{lon}",
        }
        item.update()


def get_dataset(name):
    import dtlpy as dl

    dl_auth()
    project = dl.projects.get(PROJECT)
    dataset = project.datasets.get(name)
    annotations = sum(dataset.annotations.list(), [])
    return [annotation.to_json() for annotation in annotations]


def get_geometries_from_annnotation_data(
    data: dict,
) -> List[shp.geometry.base.BaseGeometry]:
    if (len(data["annotations"]) == 1) and (
        data["annotations"][0]["label"] == "Blank",
    ):
        return []
    if isinstance(data["annotations"], str):
        rich.print(f"[yellow]{data['name']} has a url annotation")
        return []
    geometries = []
    for building_annotation in data["annotations"]:
        if (
            "coordinates" not in building_annotation
            or building_annotation["type"] == "box"
        ):
            continue
        for c in building_annotation["coordinates"]:
            if "text" in c:
                continue
            pixels = [(a["x"], a["y"]) for a in c]
            if not pixels:
                continue
            try:
                image_xy_poly = clean_building_geometry(shp.Polygon(pixels))
            except ValueError as ve:
                if "linearring requires at least 4 coordinates" in str(ve):
                    image_xy_poly = clean_building_geometry(
                        shp.box(*pixels[0], *pixels[1])
                    )
            if isinstance(image_xy_poly, shp.geometry.Polygon):
                geometries.append(image_xy_poly)
            elif isinstance(image_xy_poly, shp.geometry.MultiPolygon):
                geometries += image_xy_poly.geoms
            else:
                raise ValueError("Unexpected geometry type")
    return geometries


def main():
    import models as m
    import peewee as pw

    images = (
        m.Image.select()
        .join(
            m.PermittedLocation,
            on=pw.fn.ST_CONTAINS(m.Image.geometry, m.PermittedLocation.geometry),
        )
        .where(m.Image.label_status == "unlabeled")
        .distinct()
    )
    create_labeling_dataset("ca_labeling/ex_ante_permits", images)
