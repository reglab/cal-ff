import json
import os

import dtlpy as dl
import numpy as np
import seaborn as sns
from tqdm import tqdm

import cacafo.naip

PROJECT = "RegLab_Prod"


def dl_auth():
    if dl.token_expired():
        dl.login()


def create_labeling_dataset(prefix, images):
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
        geometry = image.geometry
        lat, lon = geometry.centroid.y, geometry.centroid.x
        item.metadata["user"] = {
            "latitude": lat,
            "longitude": lon,
            "gmaps_link": f"https://www.google.com/maps/place/{lat},{lon}&t=k",
        }
        item.update()


def get_dataset(name):
    dl_auth()
    project = dl.projects.get(PROJECT)
    dataset = project.datasets.get(name)
    annotations = sum(dataset.annotations.list(), [])
    return [annotation.to_json() for annotation in annotations]


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
