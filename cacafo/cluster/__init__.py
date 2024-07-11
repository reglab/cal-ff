import pathlib

import networkx as nx

from cacafo.db.models import Building, BuildingRelationship

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
):
    """
    Get building relationships based on the given thresholds.

    Returns a list of dictionaries with keys of BuildingRelationship fields.
    """
    return (
        BuildingRelationship.raw(
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
        .dicts()
        .execute()
    )


def _dict_of_lists(cls, **kwargs):
    dol = {
        b[0]: set() for b in Building.select(Building.id).where(Building.cafo).tuples()
    }
    query = get_building_relationships(**kwargs)
    for br in query.dicts():
        try:
            dol[br["building"]].add(br["other_building"])
        except KeyError:
            pass
    return dol


@classmethod
def building_graph(cls, **kwargs):
    """
    Return a networkx graph of the buildings and their relationships.
    kwargs are passed to get_building_relationships.
    """
    return nx.from_dict_of_lists(_dict_of_lists(**kwargs))


@classmethod
def building_clusters(cls, **kwargs):
    return nx.connected_components(cls.graph(**kwargs))
