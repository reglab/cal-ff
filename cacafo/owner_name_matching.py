import re
from functools import lru_cache

import sqlalchemy as sa
from sklearn.feature_extraction.text import TfidfVectorizer
from thefuzz import fuzz

import cacafo.db.sa_models as m
from cacafo.db.session import get_sqlalchemy_session

FUZZY_MATCH_WORDS_TO_REMOVE = [
    "llc",
    "farms",
    "trust",
    "tr",
    "trustee",
    "dairy",
    "inc",
    "revocable",
    "irrevocable",
    "farm",
    "family",
    "poultry",
    "cattle",
    "ranch",
    "acres",
    "land",
    "real",
    "estate",
    "ridge",
    "john",
    "fam",
    "partnership",
    "prop",
    "enterprises",
    "landowner",
    "lp",
    "llp",
]


def fuzzy(name_1, name_2):
    name_1 = re.sub(r"[^\w\s]", " ", name_1.lower())
    name_2 = re.sub(r"[^\w\s]", " ", name_2.lower())
    for word in FUZZY_MATCH_WORDS_TO_REMOVE:
        name_1 = name_1.replace(word, "")
        name_2 = name_2.replace(word, "")
    name_1 = " ".join(name_1.split())
    name_2 = " ".join(name_2.split())
    return fuzz.ratio(name_1, name_2) * 10


@lru_cache(maxsize=1000)
def _tfidf_vectorize(documents):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)
    documents = {d: t for d, t in zip(documents, tfidf.toarray())}
    return documents


def tf_idf(documents, name_1, name_2):
    document_arrays = _tfidf_vectorize(tuple(sorted(documents)))
    try:
        array_1 = document_arrays[name_1]
        array_2 = document_arrays[name_2]
    except KeyError:
        raise ValueError("Both names must be in the documents")
    similarity = int(array_1.dot(array_2.T) * 1000)
    return similarity


_MATCHED_OWNER_NAMES = None


def annotation(name_1, name_2):
    global _MATCHED_OWNER_NAMES
    if not _MATCHED_OWNER_NAMES:
        query = sa.select(m.ParcelOwnerNameAnnotation).where(
            m.ParcelOwnerNameAnnotation.matched
        )
        session = get_sqlalchemy_session()
        result = session.execute(query).scalars().all()
        _MATCHED_OWNER_NAMES = {r.owner_name: r.related_owner_name for r in result}
    result = _MATCHED_OWNER_NAMES.get(name_1, name_1) == name_2
    return 1000 if result else 0
