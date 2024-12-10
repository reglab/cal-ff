import json
import pandas as pd
import numpy as np
import sqlalchemy as sa

from sklearn.metrics import cohen_kappa_score
import rl 
from cacafo.db.sa_models import *
from cacafo.db.session import get_sqlalchemy_session

def load_irr_data(session):
    annotations = session.scalars(
        sa.select(IrrAnnotation)
    )
    data = []
    for row in annotations:
        is_cafo = any([x['label'] == 'cafo' for x in row.data['annotations']])
        data.append({
            'image_id':row.image_id,
            'annotator':row.annotator,
            'is_cafo': is_cafo
        })
    return pd.DataFrame(data)

def overall_cohens_kappa(data):
    data = data.sort_values('image_id')
    y1 = data['is_cafo'].iloc[::2]
    y2 = data['is_cafo'].iloc[1::2]
    assert len(y1) == len(y2), "Every image was not labeled twice"
    return cohen_kappa_score(y1, y2)

def overall_agree_pct(data):
    agree = data.groupby('image_id')['is_cafo'].nunique().eq(1).astype(int)
    return sum(agree)/len(agree)

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
            y1 = d_ij['is_cafo'].iloc[::2]
            y2 = d_ij['is_cafo'].iloc[1::2]
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
            y1 = d_ij['is_cafo'].iloc[::2]
            y2 = d_ij['is_cafo'].iloc[1::2]
            mat[i, j] = sum(y1*y2)/len(y1)
    return pd.DataFrame(mat, index = raters, columns = raters)

if __name__ == "__main__":
    session = get_sqlalchemy_session()
    data = load_irr_data(session)
    with open(rl.utils.io.get_data_path("paper", "irr_stats.txt"), 'w') as f:
        f.write(f"All rater Cohen's Kappa: {overall_cohens_kappa(data)}")
        f.write("\n")
        f.write(f"All rater agreement percent: {overall_agree_pct(data)}")
        f.write("\n")
        f.write(f"Inter-rater Cohen's Kappa Matrix:\n {cohens_kappa_matrix(data)}")
        f.write("\n")
        f.write(f"Inter-rater Agreement Matrix:\n {cohens_kappa_matrix(data)}")
        f.write("\n")