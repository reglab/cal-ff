import pathlib

import networkx as nx
import sqlalchemy as sa

from cacafo.db.sa_models import Building
from cacafo.db.session import get_sqlalchemy_session

BUILDING_THRESHOLD_RELATIONSHIP_QUERY_PATH = (
    pathlib.Path(__file__).parent / "building_threshold_relationship_query.sql"
)
BUILDING_THRESHOLD_RELATIONSHIP_QUERY = (
    BUILDING_THRESHOLD_RELATIONSHIP_QUERY_PATH.read_text()
)


def get_building_relationships(
    distance=400,
    tfidf=700,
    fuzzy=600,
    fuzzy_max=1001,
    tfidf_max=1001,
    no_owner_distance=200,
    lone_building_distance=50,
    session=None,
):
    """
    Get building relationships based on the given thresholds.

    Returns a list of dictionaries with keys of BuildingRelationship fields.
    """
    query = sa.text(
        BUILDING_THRESHOLD_RELATIONSHIP_QUERY.format(
            distance=distance,
            tfidf=tfidf,
            fuzzy=fuzzy,
            fuzzy_max=fuzzy_max,
            tfidf_max=tfidf_max,
            no_owner_distance=no_owner_distance,
            lone_building_distance=lone_building_distance,
        )
    )
    if session is None:
        session = get_sqlalchemy_session()
    return session.execute(query).fetchall()


def dict_of_lists(session=None, drop_excluded_buildings=True, **kwargs):
    if session is None:
        session = get_sqlalchemy_session()
    query = get_building_relationships(session=session, **kwargs)
    condition = True
    if drop_excluded_buildings:
        condition = Building.excluded_at.is_(None)
    included_ids = set(
        building.id
        for building in session.execute(sa.select(Building).where(condition))
        .scalars()
        .all()
    )
    dol = {id_: [] for id_ in included_ids}
    for building_id, related_building_id in query:
        if building_id not in dol or related_building_id not in dol:
            continue
        dol[building_id].append(related_building_id)
    return dol


def building_graph(**kwargs):
    """
    Return a networkx graph of the buildings and their relationships.
    kwargs are passed to get_building_relationships.
    """
    return nx.from_dict_of_lists(dict_of_lists(**kwargs))


def building_clusters(**kwargs):
    return nx.connected_components(building_graph(**kwargs))
