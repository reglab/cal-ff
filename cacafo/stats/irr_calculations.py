import json
import pandas as pd
import numpy as np
import sqlalchemy as sa

from sklearn.metrics import cohen_kappa_score
import rl 
from cacafo.db.sa_models import *
from cacafo.db.session import get_sqlalchemy_session

import cacafo.query as query

def load_irr_data(session):
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

    image_ids = {
        'positive': [x.id for x in positive_images],
        'labeled negative': [x.id for x in labeled_negative_images],
        'high confidence negative': [x.id for x in high_confidence_negative_images],
        'low confidence negative': [x.id for x in low_confidence_negative_images]
    }
    annotations = session.scalars(
        sa.select(IrrAnnotation)
    )
    data = []
    for row in annotations:
        is_cafo = any([x['label'] == 'cafo' for x in row.data['annotations']])
        image_category = [x for x in image_ids if row.image_id in image_ids[x]][0]
        data.append({
            'image_id':row.image_id,
            'annotator':row.annotator,
            'is_cafo': is_cafo,
            'image_category': image_category
        })
    return pd.DataFrame(data)

def overall_cohens_kappa(data):
    data = data.sort_values('image_id')
    y1 = data['is_cafo'].iloc[::2].array
    y2 = data['is_cafo'].iloc[1::2].array
    assert len(y1) == len(y2), "Every image was not labeled twice"
    return cohen_kappa_score(y1, y2)

def overall_agree_pct(data):
    data = data.sort_values('image_id')
    y1 = data['is_cafo'].iloc[::2].array
    y2 = data['is_cafo'].iloc[1::2].array
    return sum(y1 == y2)/(len(y1)+len(y2))

def cohens_kappa_matrix(data):
    raters = data['annotator'].unique()
    mat = np.ones([len(raters), len(raters)])
    for i in range(len(raters)):
        d_i = data.groupby('image_id').filter(lambda x: any(x['annotator'] == raters[i]))
        for j in range(len(raters)):
            if i == j:
                continue
            d_ij = d_i.groupby('image_id').filter(lambda x: any(x['annotator'] == raters[j]))
            d_ij = d_ij.sort_values('image_id')
            y1 = d_ij['is_cafo'].iloc[::2].array
            y2 = d_ij['is_cafo'].iloc[1::2].array
            mat[i, j] = cohen_kappa_score(y1, y2)
    return pd.DataFrame(mat, index = raters, columns = raters)

def agree_pct_matrix(data):
    raters = data['annotator'].unique()
    mat = np.ones([len(raters), len(raters)])
    for i in range(len(raters)):
        d_i = data.groupby('image_id').filter(lambda x: any(x['annotator'] == raters[i]))
        for j in range(len(raters)):
            if i == j:
                continue
            d_ij = d_i.groupby('image_id').filter(lambda x: any(x['annotator'] == raters[j]))
            d_ij = d_ij.sort_values('image_id')
            y1 = d_ij['is_cafo'].iloc[::2].array
            y2 = d_ij['is_cafo'].iloc[1::2].array
            mat[i, j] = sum(y1 == y2)/(len(y1)+len(y2))
    return pd.DataFrame(mat, index = raters, columns = raters)

def stats_by_category(data):
    categories = data['image_category'].unique()
    kappas = []
    agrees = []
    totals = 0
    for c in categories:
        d_c = data[data['image_category'] == c]
        d_c = d_c.sort_values('image_id')
        y1 = d_c['is_cafo'].iloc[::2].array
        y2 = d_c['is_cafo'].iloc[1::2].array
        assert len(y1)==len(y2)
        kappas.append(cohen_kappa_score(y1, y2))
        agrees.append(sum(y1 == y2)/(len(y1)+len(y2)))
        totals+=len(y1)+len(y2)
    return pd.DataFrame({'image category':categories, 
                         "Cohen's Kappa": kappas, 
                         "Agreement percent": agrees},
                         ).set_index('image category')
if __name__ == "__main__":
    session = get_sqlalchemy_session()
    data = load_irr_data(session)
    with open(rl.utils.io.get_data_path("paper", "irr_stats.txt"), 'w') as f:
        f.write(f"All rater Cohen's Kappa: {overall_cohens_kappa(data)}")
        f.write("\n")
        f.write(f"All rater agreement percent: {overall_agree_pct(data)}")
        f.write("\n")
        f.write(f"Inter-rater Cohen's Kappa matrix:\n {cohens_kappa_matrix(data)}")
        f.write("\n")
        f.write(f"Inter-rater agreement matrix:\n {cohens_kappa_matrix(data)}")
        f.write("\n")
        f.write(f"Stats by selection category:\n {stats_by_category(data)}")
        f.write("\n")