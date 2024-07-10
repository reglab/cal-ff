from datetime import datetime

import sshtunnel

from models import *


def create_saskia_csv():
    facility_id_to_county = {
        row["id"]: County.geocode(lon=row["longitude"], lat=row["latitude"]).name
        for row in Facility.select(
            Facility.id, Facility.longitude, Facility.latitude
        ).dicts()
    }
    any_equals = lambda name: pw.fn.COALESCE(
        pw.fn.MAX((AnimalType.name == name).cast("int")), 0
    )
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
            any_equals("cows").alias("cows"),
            any_equals("pigs").alias("pigs"),
            any_equals("poultry").alias("poultry"),
            any_equals("sheep").alias("sheep"),
            any_equals("horses").alias("horses"),
            any_equals("goats").alias("goats"),
            any_equals("auction").alias("auction"),
            any_equals("dairy").alias("dairy"),
            any_equals("calves").alias("calves"),
        )
        .join(ConstructionAnnotation)
        .join(
            FacilityAnimalType,
            on=(FacilityAnimalType.facility == Facility.id),
            join_type=pw.JOIN.LEFT_OUTER,
        )
        .join(
            AnimalType,
            on=(FacilityAnimalType.animal_type == AnimalType.id),
            join_type=pw.JOIN.LEFT_OUTER,
        )
        .group_by(Facility.id, ConstructionAnnotation.id)
        .dicts()
    )
    rows = list(rows)
    rows = [{"county": facility_id_to_county[r["facility_id"]], **r} for r in rows]
    date = datetime.now().strftime("%Y-%m-%d")
    writer = csv.DictWriter(
        open(f"outputs/construction_dates_temp_{date}.csv", "w"),
        fieldnames=rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(rows)
    return rows


if __name__ == "__main__":
    create_saskia_csv()
