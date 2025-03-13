import random

import numpy as np
import pandas as pd
import rl
import sklearn.metrics as metrics
import sqlalchemy as sa

import cacafo.db.models as m
import cacafo.query as query
from cacafo.db.session import new_session

_IRR_DATA = None


def image_category_map(session):
    positive_images = session.execute(query.positive_images()).scalars().all()
    labeled_negative_images = (
        session.execute(query.labeled_negative_images()).scalars().all()
    )
    high_confidence_negative_images = (
        session.execute(query.high_confidence_negative_images()).scalars().all()
    )
    low_confidence_negative_images = (
        session.execute(query.low_confidence_negative_images()).scalars().all()
    )
    return (
        {x.id: "positive" for x in positive_images}
        | {x.id: "labeled negative" for x in labeled_negative_images}
        | {x.id: "high confidence negative" for x in high_confidence_negative_images}
        | {x.id: "low confidence negative" for x in low_confidence_negative_images}
    )


def image_id_map(session):
    positive_images = session.execute(query.positive_images()).scalars().all()
    labeled_negative_images = (
        session.execute(query.labeled_negative_images()).scalars().all()
    )
    high_confidence_negative_images = (
        session.execute(query.high_confidence_negative_images()).scalars().all()
    )
    low_confidence_negative_images = (
        session.execute(query.low_confidence_negative_images()).scalars().all()
    )
    return {
        x.id: x
        for x in positive_images
        + labeled_negative_images
        + high_confidence_negative_images
        + low_confidence_negative_images
    }


def load_irr_data(session, cache=True):
    global _IRR_DATA
    if cache and _IRR_DATA is not None:
        return _IRR_DATA

    _image_id_map = image_id_map(session)
    _image_category_map = image_category_map(session)

    annotations = session.scalars(
        sa.select(m.IrrAnnotation).join(m.Image).where(m.Image.bucket.is_not(None))
    )
    data = []
    for row in annotations:
        is_cafo = any([x["label"] == "cafo" for x in row.data["annotations"]])
        image_category = _image_category_map[row.image_id]
        data.append(
            {
                "image_id": row.image_id,
                "annotator": row.annotator,
                "is_cafo": is_cafo,
                "image_category": image_category,
                "bucket": _image_id_map[row.image_id].bucket,
            }
        )
    _IRR_DATA = pd.DataFrame(data)
    return _IRR_DATA


def pairwise_irr_data(session, cache=True):
    df = load_irr_data(session, cache)
    df = df.sort_values("image_id")
    assert (
        df["image_id"].iloc[::2].array == df["image_id"].iloc[1::2].array
    ), "Image ids unaligned"
    y1 = df["is_cafo"].iloc[::2].array
    y2 = df["is_cafo"].iloc[1::2].array
    y1_name = df["annotator"].iloc[::2].array
    y2_name = df["annotator"].iloc[1::2].array
    return pd.DataFrame(
        {
            "image_id": df["image_id"].iloc[::2].array,
            "annotator_1": y1_name,
            "annotator_2": y2_name,
            "is_cafo_1": y1,
            "is_cafo_2": y2,
            "image_category": df["image_category"].iloc[::2].array,
            "bucket": df["bucket"].iloc[::2].array,
        }
    )


def overall_cohens_kappa(session):
    df = pairwise_irr_data(session)
    return metrics.cohen_kappa_score(df["is_cafo_1"], df["is_cafo_2"])


def overall_agree_pct(session):
    df = pairwise_irr_data(session)
    return sum(df["is_cafo_1"] == df["is_cafo_2"]) / len(df)


def cohens_kappa_matrix(session):
    data = load_irr_data(session)
    raters = data["annotator"].unique()
    mat = np.ones([len(raters), len(raters)])
    for i in range(len(raters)):
        d_i = data.groupby("image_id").filter(
            lambda x: any(x["annotator"] == raters[i])
        )
        for j in range(len(raters)):
            if i == j:
                continue
            d_ij = d_i.groupby("image_id").filter(
                lambda x: any(x["annotator"] == raters[j])
            )
            d_ij = d_ij.sort_values("image_id")
            assert (
                d_ij["image_id"].iloc[::2].array == d_ij["image_id"].iloc[1::2].array
            ), f"Image ids unlaigned, {i, j}"
            y1 = d_ij["is_cafo"].iloc[::2].array
            y2 = d_ij["is_cafo"].iloc[1::2].array
            mat[i, j] = metrics.cohen_kappa_score(y1, y2)
    return pd.DataFrame(mat, index=raters, columns=raters)


def agree_pct_matrix(session):
    data = load_irr_data(session)
    raters = data["annotator"].unique()
    mat = np.ones([len(raters), len(raters)])
    for i in range(len(raters)):
        d_i = data.groupby("image_id").filter(
            lambda x: any(x["annotator"] == raters[i])
        )
        for j in range(len(raters)):
            if i == j:
                continue
            d_ij = d_i.groupby("image_id").filter(
                lambda x: any(x["annotator"] == raters[j])
            )
            d_ij = d_ij.sort_values("image_id")
            assert (
                d_ij["image_id"].iloc[::2].array == d_ij["image_id"].iloc[1::2].array
            ), f"Image ids unlaigned, {i, j}"
            y1 = d_ij["is_cafo"].iloc[::2].array
            y2 = d_ij["is_cafo"].iloc[1::2].array
            mat[i, j] = sum(y1 == y2) / (len(y1))
    return pd.DataFrame(mat, index=raters, columns=raters)


def agreement_report(y1, y2):
    return {
        "cohens_kappa": metrics.cohen_kappa_score(y1, y2),
        "n": len(y1),
        "n_agree_true": sum(y1 & y2),
        "n_agree_false": sum(~y1 & ~y2),
        "n_disagree": sum(y1 != y2),
        "agreement_pct": sum(y1 == y2) / len(y1),
    }


def stats_by_category(session):
    df = pairwise_irr_data(session)
    categories = df["image_category"].unique()
    return pd.DataFrame.from_records(
        [
            {
                "image_category": c,
            }
            | agreement_report(
                df[df["image_category"] == c]["is_cafo_1"],
                df[df["image_category"] == c]["is_cafo_2"],
            )
            for c in categories
        ]
    ).set_index("image_category")


def stats_by_bucket(session):
    df = pairwise_irr_data(session)
    buckets = df["bucket"].unique()
    return pd.DataFrame.from_records(
        [
            {
                "bucket": c,
            }
            | agreement_report(
                df[df["bucket"] == c]["is_cafo_1"], df[df["bucket"] == c]["is_cafo_2"]
            )
            for c in buckets
        ]
    ).set_index("bucket")


def stats_by_label_status(session):
    bucket_map = {
        "1": "1",
        "0": "0",
        1: "1",
        0: "0",
    }
    data = pairwise_irr_data(session)
    data["label_status"] = data["bucket"].apply(
        lambda x: bucket_map.get(x, "fully labeled")
    )
    label_statuses = data["label_status"].unique()
    return pd.DataFrame.from_records(
        [
            {
                "label_status": c,
            }
            | agreement_report(
                data[data["label_status"] == c]["is_cafo_1"],
                data[data["label_status"] == c]["is_cafo_2"],
            )
            for c in label_statuses
        ]
    ).set_index("label_status")


def label_balanced_cohens_kappa(session):
    class_counts = {
        "positive": 0,
        "labeled negative": 0,
        "high confidence negative": 0,
        "low confidence negative": 0,
    }
    map = image_category_map(session)
    class_counts.update(
        {v: sum(w == v for w in map.values()) for v in class_counts.keys()}
    )
    df = pairwise_irr_data(session)
    pairs_by_class = {k: df[df["image_category"] == k] for k in class_counts.keys()}

    random.seed(52)
    # sample pairs proportional to class counts
    sample = []
    for k, v in pairs_by_class.items():
        sample += v.sample(n=class_counts[k], replace=True).to_dict(orient="records")
    sample_df = pd.DataFrame(sample)
    return metrics.cohen_kappa_score(sample_df["is_cafo_1"], sample_df["is_cafo_2"])


def label_balanced_agreement_pct(session):
    class_counts = {
        "positive": 0,
        "labeled negative": 0,
        "high confidence negative": 0,
        "low confidence negative": 0,
    }
    map = image_category_map(session)
    class_counts.update(
        {v: sum(w == v for w in map.values()) for v in class_counts.keys()}
    )
    df = pairwise_irr_data(session)
    pairs_by_class = {k: df[df["image_category"] == k] for k in class_counts.keys()}

    random.seed(52)
    # sample pairs proportional to class counts
    sample = []
    for k, v in pairs_by_class.items():
        sample += v.sample(n=class_counts[k], replace=True).to_dict(orient="records")
    sample_df = pd.DataFrame(sample)
    return sum(sample_df["is_cafo_1"] == sample_df["is_cafo_2"]) / len(sample_df)


if __name__ == "__main__":
    session = new_session()
    data = load_irr_data(session)
    with open(rl.utils.io.get_data_path("paper", "irr_stats.txt"), "w") as f:
        f.write(f"All rater Cohen's Kappa: {overall_cohens_kappa(session)}")
        f.write("\n")
        f.write(f"All rater agreement percent: {overall_agree_pct(session)}")
        f.write("\n")
        f.write(f"Inter-rater Cohen's Kappa matrix:\n {cohens_kappa_matrix(session)}")
        f.write("\n")
        f.write(f"Inter-rater agreement matrix:\n {agree_pct_matrix(session)}")
        f.write("\n")
        f.write(f"Stats by selection category:\n {stats_by_category(session)}")
        f.write("\n")
