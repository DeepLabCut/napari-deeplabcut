import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from napari_deeplabcut._writer import _form_df
from napari_deeplabcut.misc import DLCHeader


def _cluster(data):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)

    # putting components in a dataframe for later
    PCA_components = pd.DataFrame(principalComponents)

    dbscan=DBSCAN(eps=9.7, min_samples=20, algorithm='ball_tree', metric='minkowski', leaf_size=90, p=2)

    # fit - perform DBSCAN clustering from features, or distance matrix.
    dbscan = dbscan.fit(PCA_components)
    cluster1 = dbscan.labels_

    return PCA_components, cluster1


def cluster_data(points_layer):
    df = _form_df(
        points_layer.data,
        dict(metadata=points_layer.metadata, properties=points_layer.properties),
    )

    try:
        df = df.drop('single', axis=1, level='individuals')
    except KeyError:
        pass
    df.dropna(inplace=True)
    header = DLCHeader(df.columns)
    try:
        df = df.stack('individuals').droplevel('individuals')
    except KeyError:
        pass

    images = []
    for img in df.index:
        if isinstance(img, (str, int)):
            images.append(img)
        else:
            images.append("/".join(img))

    df_xy = df.iloc[:, df.columns.get_level_values("coords").isin(["x", "y"])]

    xy = df_xy.to_numpy().reshape((-1, len(header.bodyparts), 2))
    # TODO Normalize dists by longest length?
    dists = np.vstack([pdist(data, "euclidean") for data in xy])
    points = np.c_[_cluster(dists)]  # x, y, label
    return points, images
