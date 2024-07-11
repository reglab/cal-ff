import peewee as pw
import rich
import rich_click as click

import cacafo.db.models as m

checks = {}


def check(expected=None):
    def wrapper(func):
        checks[func] = expected

    return wrapper


@check(expected=0)
def facilities_without_animal_types():
    return (
        m.Facility.select()
        .join(m.FacilityAnimalType, pw.JOIN.LEFT_OUTER)
        .where(m.FacilityAnimalType.id.is_null())
        .count()
    )


@check()
def permits_without_locations():
    return (
        m.Permit.select()
        .join(m.PermitPermittedLocation)
        .where(m.PermitPermittedLocation.id.is_null())
        .count()
    )


@check()
def facilities_without_any_permits():
    facilities = (
        m.Facility.select()
        .join(m.FacilityPermittedLocation, pw.JOIN.LEFT_OUTER)
        .where(m.FacilityPermittedLocation.id.is_null())
    )
    return facilities.count()


@check()
def num_facilities():
    return m.Facility.select().count()


@click.command()
def check():
    for func, expected in checks.items():
        result = func()
        name = func.__name__.replace("_", " ")
        if expected is not None and result != expected:
            rich.print(f"[red]Error[/red]: {name}: {result} != {expected}")
        else:
            rich.print(f"[green]OK[/green] {name}: {result}")
