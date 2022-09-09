import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from napari_deeplabcut._writer import _conv_layer_to_df
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
    df = _conv_layer_to_df(
        points_layer.data, points_layer.metadata, points_layer.properties
    )
    df.dropna(inplace=True)
    df.index = ['/'.join(row) for row in list(df.index)]
    header = DLCHeader(df.columns)
    xy = df.to_numpy().reshape((-1, len(header.bodyparts), 2))
    # TODO Normalize dists by longest length?
    dists = np.vstack([pdist(data, "euclidean") for data in xy])
    points = np.c_[_cluster(dists)]  # x, y, label
    return points, list(df.index)
