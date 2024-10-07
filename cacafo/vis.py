import io
from typing import List, Tuple

import geoalchemy2 as ga
import matplotlib.pyplot as plt
import rich_click as click
import sqlalchemy as sa
from PIL import Image
from shapely.geometry import Point
from sqlalchemy.orm import Session

from cacafo.dataloop import get_geometries_from_annnotation_data
from cacafo.db.sa_models import Image as DBImage
from cacafo.db.sa_models import ImageAnnotation
from cacafo.naip import Format, download_ca_cafo_naip_image
from cacafo.transform import DEFAULT_SRID


def visualize_annotations(image: DBImage, annotations: List[ImageAnnotation], ax=None):
    """
    Visualize an annotation on the corresponding image.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # Download and display the image
    img_data = download_ca_cafo_naip_image(image.name, Format.JPEG)
    img = Image.open(io.BytesIO(img_data.getvalue()))
    ax.imshow(img)

    # Process and display annotations
    for annotation in annotations:
        geometries = get_geometries_from_annnotation_data(annotation.data)
        for geometry in geometries:
            ax.plot(
                *geometry.exterior.xy,
                marker="o",
                markersize=1,
                linestyle="-",
                color="red",
            )
            centroid = geometry.centroid
            ax.annotate(
                str(annotation.id),
                (centroid.x, centroid.y),
                color="red",
                fontsize=8,
                ha="center",
                va="center",
            )

    ax.set_title(f"Image: {image.name}")
    ax.axis("off")
    return ax


def get_image(
    session: Session, image_name: str = None, lat: float = None, lon: float = None
) -> Tuple[DBImage, List[ImageAnnotation]]:
    """
    Retrieve the image and its latest annotation based on image name or geographic location.
    """
    if image_name:
        query = sa.select(DBImage).where(DBImage.name == image_name)
    elif lat is not None and lon is not None:
        point = Point(lon, lat)
        # Assuming DBImage.geometry is in SRID 4326 (WGS84)
        query = sa.select(DBImage).where(
            sa.func.ST_Contains(
                sa.cast(DBImage.geometry, ga.Geometry("GEOMETRY", srid=DEFAULT_SRID)),
                sa.func.ST_SetSRID(ga.shape.from_shape(point), DEFAULT_SRID),
            )
            & (DBImage.bucket.is_not(None))
        )
    else:
        raise ValueError("Either image_name or both lat and lon must be provided")

    image = session.execute(query).scalar_one_or_none()
    if not image:
        raise ValueError("No image found for the given criteria")

    return image


@click.group("vis", help="Commands for visualizing data")
def _cli():
    pass


@_cli.command(
    "image-annotation", help="Visualize annotations for a given image or location"
)
@click.option("--image-name", help="Name of the image to visualize")
@click.option("--lat", type=float, help="Latitude of the location")
@click.option("--lon", type=float, help="Longitude of the location")
@click.option("--output", type=click.Path(), help="Path to save the output image")
def image_annotation(image_name: str, lat: float, lon: float, output: str):
    """
    Visualize annotations for a given image or location.
    """
    from cacafo.db.session import get_sqlalchemy_session

    with get_sqlalchemy_session() as session:
        try:
            image = get_image(session, image_name, lat, lon)
            if not image.annotations:
                click.echo("No annotation found for the given image.")
                return

            fig, ax = plt.subplots(figsize=(12, 12))
            visualize_annotations(image, image.annotations, ax)

            if output:
                plt.savefig(output, bbox_inches="tight")
                click.echo(f"Visualization saved to {output}")
            else:
                plt.show()

        except ValueError as e:
            click.echo(f"Error: {str(e)}")
