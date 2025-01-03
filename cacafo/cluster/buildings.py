import pathlib

import networkx as nx
import sqlalchemy as sa

import cacafo.db.models as m
from cacafo.db.session import new_session

BUILDING_THRESHOLD_RELATIONSHIP_QUERY_PATH = (
    pathlib.Path(__file__).parent / "building_threshold_relationship_query.sql"
)
BUILDING_THRESHOLD_RELATIONSHIP_QUERY = (
    BUILDING_THRESHOLD_RELATIONSHIP_QUERY_PATH.read_text()
)


class MemoryBuildingRelationships:
    """
    In order to test a number of thresholds quickly,
    we can load all building relationships into memory and query
    them there
    """

    tree = None
    no_owner_building_ids = set()
    all_building_ids = set()

    def __init__(self, session=None):
        self.no_owner_building_ids = set(
            session.scalars(
                sa.select(m.Building.id)
                .join(m.Parcel, isouter=True)
                .where(m.Parcel.owner.is_(None) | (m.Parcel.owner == ""))
            ).all()
        )

        self.all_building_ids = set(session.scalars(sa.select(m.Building.id)).all())

        rows = session.execute(
            sa.text("SELECT * FROM building_relationship")
        ).fetchall()
        # row is (id, reason, weight, building_id, related_building_id)
        self.tree = {}
        for row in rows:
            reason = row[1]
            weight = row[2]
            if reason not in self.tree:
                self.tree[reason] = {}
            if weight not in self.tree[reason]:
                self.tree[reason][weight] = set()
            self.tree[reason][weight].add((row[3], row[4]))

    def get_reason_range(self, reason, min_weight, max_weight):
        if reason not in self.tree:
            return set()
        weights = range(min_weight, max_weight)
        return set.union(*(self.tree[reason].get(weight, set()) for weight in weights))

    def threshold_relationship_query(
        self,
        distance=400,
        tfidf=700,
        fuzzy=600,
        fuzzy_max=1001,
        tfidf_max=1001,
        no_owner_distance=200,
        lone_building_distance=50,
    ):
        threshold_relationships = set.intersection(
            self.get_reason_range("distance", 1000 - distance, 1001),
            self.get_reason_range("tfidf", tfidf, tfidf_max),
            self.get_reason_range("fuzzy", fuzzy, fuzzy_max),
        )
        parcel_name_annotation_relationships = set.intersection(
            self.get_reason_range("distance", 1000 - distance, 1001),
            self.get_reason_range("parcel owner annotation", 999, 1001),
        )
        no_owner_relationships = {
            rel
            for rel in self.get_reason_range("distance", 1000 - no_owner_distance, 1001)
            if rel[0] in self.no_owner_building_ids
            or rel[1] in self.no_owner_building_ids
        }
        matching_parcel_relationships = self.get_reason_range(
            "matching parcel", 0, 1001
        )
        all_relationships = (
            threshold_relationships
            | no_owner_relationships
            | matching_parcel_relationships
            | parcel_name_annotation_relationships
        )
        # in theory all should have matching both sides so don't need to get rel[1]
        lone_buildings = self.all_building_ids - {rel[0] for rel in all_relationships}
        lone_building_relationships = {
            rel
            for rel in self.get_reason_range(
                "distance", 1000 - lone_building_distance, 1001
            )
            if rel[0] in lone_buildings or rel[1] in lone_buildings
        }
        return all_relationships | lone_building_relationships


_MEMORY_BUILDING_RELATIONSHIPS = None


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
    global _MEMORY_BUILDING_RELATIONSHIPS
    if _MEMORY_BUILDING_RELATIONSHIPS is None:
        _MEMORY_BUILDING_RELATIONSHIPS = MemoryBuildingRelationships(session=session)
    return _MEMORY_BUILDING_RELATIONSHIPS.threshold_relationship_query(
        distance=distance,
        tfidf=tfidf,
        fuzzy=fuzzy,
        fuzzy_max=fuzzy_max,
        tfidf_max=tfidf_max,
        no_owner_distance=no_owner_distance,
        lone_building_distance=lone_building_distance,
    )


def dict_of_lists(session=None, drop_excluded_buildings=True, **kwargs):
    if session is None:
        session = new_session()
    query = get_building_relationships(session=session, **kwargs)
    condition = True
    if drop_excluded_buildings:
        condition = m.Building.excluded_at.is_(None)
    included_ids = set(
        building.id
        for building in session.execute(sa.select(m.Building).where(condition))
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
