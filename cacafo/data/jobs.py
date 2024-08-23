import peewee as pw
import rich
import rich_click as click

import cacafo.db.models as m

jobs = {}


def job(func):
    jobs[func.__name__] = func
    return func


@job
def propagate_not_afo_cafo():
    """
    Propagate not afo and not cafo animal types to underlying
    Buildings. delete facilities marked with these types.
    """

    fa = FacilityArchive.insert_from(
                                    query=Facility.select())

    facility_ids = (
        m.Facility.select(m.Facility.id)
        .join(m.FacilityAnimalType)
        .join(m.AnimalType)
        .where((m.AnimalType.name == "not afo") | (m.AnimalType.name == "not cafo"))
    ).tuples()
    facility_ids = [f[0] for f in facility_ids]
    n_buildings = (
        m.Building.update(cafo=False)
        .where(m.Building.facility_id.in_(facility_ids))
        .execute()
    )
    n_buildings = (
        m.Building.update(facility_id=None)
        .where(m.Building.facility_id.in_(facility_ids))
        .execute()
    )

    n_fat = (
        m.FacilityAnimalType.delete()
        .where(m.FacilityAnimalType.facility_id.in_(facility_ids))
        .execute()
    )
    n_ca = (
        m.ConstructionAnnotation.update(facility_id=None)
        .where(m.ConstructionAnnotation.facility_id.in_(facility_ids))
        .execute()
    )
    n_fpl = (
        m.FacilityPermittedLocation.delete()
        .where(m.FacilityPermittedLocation.facility_id.in_(facility_ids))
        .execute()
    )
    n_facilities = m.Facility.delete().where(m.Facility.id.in_(facility_ids)).execute()
    n_permits = (
        m.Permit.update(facility_id=None)
        .where(m.Permit.facility_id.in_(facility_ids))
        .execute()
    )
    rich.print(f"Deleted {n_facilities} facilities and updated {n_buildings} buildings")
    rich.print(
        f"Deleted {n_fat} FacilityAnimalType rows, {n_ca} ConstructionAnnotation rows, {n_fpl} FacilityPermittedLocation rows, and {n_permits} Permit rows."
    )
    return n_facilities, n_buildings


@click.command("jobs")
@click.argument("job", type=click.Choice(jobs.keys()), required=True)
def cmd_jobs(job):
    if job in jobs:
        jobs[job]()
    else:
        rich.print(f"Job {job} not found")
        return 1
