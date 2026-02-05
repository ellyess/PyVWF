"""Clustering utilities for turbine metadata.

This module provides spatial clustering of turbine coordinates for training and
evaluation workflows.
"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def cluster_turbines(num_clu, turb_info_train, train=False, *args):
    """Cluster turbines by spatial coordinates.

    Args:
        num_clu (int): Number of clusters.
        turb_info_train (pandas.DataFrame): Training turbine metadata with ``lat`` and ``lon``.
        train (bool): If True, assign clusters to ``turb_info_train`` and return it.
        *args (pandas.DataFrame): Optional turbine metadata to cluster using the fitted model.

    Returns:
        pandas.DataFrame: Turbine metadata with an added ``cluster`` column.
    """
    # fitting clusters to training data
    kmeans = KMeans(
            init="random",
            n_clusters = num_clu,
            n_init = 10,
            max_iter = 300,
            random_state = 42
        )
    kmeans.fit(turb_info_train[['lat','lon']])
        
    if train == True:
        turb_info_train['cluster'] = kmeans.predict(turb_info_train[['lat','lon']])
        return turb_info_train
    else:
        turb_info = args[0]
        turb_info['cluster'] = kmeans.predict(turb_info[['lat','lon']])
        return turb_info
