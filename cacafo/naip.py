from enum import Enum
from io import BytesIO

import contextily as ctx
import diskcache as dc
import rasterio
import rasterio.merge
import rich_click as click
import shapely as shp
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from rasterio.crs import CRS
from tqdm import tqdm

CA_2016_COUNTIES = [
    "Fresno",
    "Kings",
    "Madera",
    "Monterey",
    "San Benito",
    "San Luis Obispo",
    "Tulare",
]
BUCKET_NAME = "image-hub"
IMAGE_DIR_FORMAT = "NAIP-RGB-masked/{ca_dir}/{county}/{subdir}/"
IMAGE_NAME_FORMAT = IMAGE_DIR_FORMAT + "{image_name}.{format_}"


dc_cache = dc.Cache(".naip_cache")


class Format(Enum):
    TIF = "tif"
    JPEG = "jpeg"


def _storage_client():
    try:
        return storage.Client()
    except DefaultCredentialsError:
        raise DefaultCredentialsError(
            "No credentials found. Try running 'gcloud auth application-default login' in your terminal and then re-running this method."
        )


@dc_cache.memoize()
def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    storage_client = _storage_client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    # get blob as PIL Image
    content = blob.download_as_bytes()
    return BytesIO(content)


def get_naip_image_path(image_name, format_: Format = Format.JPEG):
    county = image_name.split("_")[0]
    ca_dir = "CA_2016" if county in CA_2016_COUNTIES else "CA"
    image_path = IMAGE_NAME_FORMAT.format(
        ca_dir=ca_dir,
        county=county,
        image_name=image_name,
        subdir="images" if format_ == Format.JPEG else "tiled_tif",
        format_=format_.value,
    )
    return image_path


def download_ca_cafo_naip_image(image_name, format_: Format = Format.JPEG):
    image_path = get_naip_image_path(image_name, format_)
    return download_blob(BUCKET_NAME, image_path)


def add_basemap(ax):
    from cacafo.db.session import new_session

    session = new_session()

    @dc_cache.memoize()
    def get_images_for_area(xmin, ymin, xmax, ymax):
        from cacafo.db.models import Image as mImage

        nonlocal session
        return mImage.get_images_for_area(
            shp.Polygon.from_bounds(xmin, ymin, xmax, ymax), session
        )

    xmin, xmax, ymin, ymax = ax.axis()
    # project from 3311 to 4326 if necessary
    if abs(xmin) > 180:
        xmin, ymin = rasterio.warp.transform("EPSG:3311", "EPSG:4326", [xmin], [ymin])
        xmax, ymax = rasterio.warp.transform("EPSG:3311", "EPSG:4326", [xmax], [ymax])
        xmin, ymin, xmax, ymax = xmin[0], ymin[0], xmax[0], ymax[0]

    images = get_images_for_area(xmin, ymin, xmax, ymax)
    names = [image.name for image in images]
    if not names:
        print(f"No images found for area {xmin, ymin, xmax, ymax}")
        return ax
    datasets = []
    for name in tqdm(names, desc="Downloading images"):
        try:
            datasets.append(download_ca_cafo_naip_image(name, format_=Format.TIF))
        except Exception as e:
            if "No such object" in str(e):
                print(f"Image {name} not found.")
            else:
                raise e
    rasterio_datasets = [rasterio.open(dataset) for dataset in datasets]
    output_file = BytesIO()
    rasterio.merge.merge(rasterio_datasets, dst_path=output_file)
    ctx.add_basemap(
        ax,
        source=output_file,
        crs=CRS.from_epsg(3311) if abs(ax.axis()[0]) > 180 else CRS.from_epsg(4326),
    )
    return ax


def list_available_images():
    from cacafo.db.models import County

    counties = sum(County.select(County.name).tuples(), ())
    client = _storage_client()
    for county in counties:
        prefix = IMAGE_DIR_FORMAT.format(
            ca_dir="CA_2016" if county in CA_2016_COUNTIES else "CA",
            county=county,
            subdir="images",
        )
        names = dc_cache.get(f"available_images_{county}")
        names = (
            blob.name.split("/")[-1].split(".")[0]
            for blob in client.list_blobs(BUCKET_NAME, prefix=f"{prefix}")
        )
        images = []
        for name in names:
            images.append(name)
            yield name
        dc_cache.set(f"available_images_{county}", images)


def list_removed_images():
    from cacafo.db.models import County

    counties = sum(County.select(County.name).tuples(), ())
    client = _storage_client()
    for county in counties:
        prefix = IMAGE_DIR_FORMAT.format(
            ca_dir="CA_2016" if county in CA_2016_COUNTIES else "CA",
            county=county,
            subdir="black_imgs",
        )
        names = dc_cache.get(f"removed_images_{county}")
        if not names:
            names = (
                blob.name.split("/")[-1].split(".")[0]
                for blob in client.list_blobs(BUCKET_NAME, prefix=f"{prefix}")
            )
        images = []
        for name in names:
            images.append(name)
            yield name
        dc_cache.set(f"removed_images_{county}", images)


def list_images_with_prefix(prefix):
    client = _storage_client()
    names = (
        blob.name.split("/")[-1].split(".")[0]
        for blob in client.list_blobs(BUCKET_NAME, prefix=f"{prefix}")
    )
    for name in names:
        yield name


def copy_blob(
    bucket_name, source_blob_name, destination_bucket_name, destination_blob_name
):
    """Copies a blob from one bucket to another with a new name."""
    storage_client = _storage_client()
    source_bucket = storage_client.get_bucket(bucket_name)
    source_blob = source_bucket.blob(source_blob_name)
    destination_bucket = storage_client.get_bucket(destination_bucket_name)
    source_bucket.copy_blob(source_blob, destination_bucket, destination_blob_name)


def create_subset(prefix, image_names, format_: Format = Format.JPEG):
    _client = _storage_client()
    prefix = prefix.strip("/")
    for image in tqdm(image_names):
        image_path = get_naip_image_path(image, format_)
        copy_blob(
            BUCKET_NAME, image_path, BUCKET_NAME, f"{prefix}/{image}.{format_.value}"
        )


@click.group("naip", help="Commands for working with NAIP images on GCP")
def _cli():
    pass


@_cli.group("ls", help="List available or removed images.")
def _ls():
    pass


@_ls.command("available")
def _ls_available():
    for image in list_available_images():
        print(image)


@_ls.command("removed")
def _ls_removed():
    for image in list_removed_images():
        print(image)


@_cli.command("dl", help="Download an image from GCP.")
@click.option("--image-name", required=True)
@click.option("--output-path", required=True, type=click.Path())
@click.option("--format", type=click.Choice(["jpeg", "tif"]), default="jpeg")
def _download(image_name, output_path, format):
    format_ = Format.JPEG if format == "jpeg" else Format.TIF
    image = download_ca_cafo_naip_image(image_name, format_)
    with open(output_path, "wb") as f:
        f.write(image.read())
