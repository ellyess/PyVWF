from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def cluster_turbines(num_clu, turb_info_train, train=False, *args):
    """
    Spatially cluster the turbines using the coordinates.

        Args:
            num_clu (int): number of clusters to split turbines into
            turb_train (pandas.DataFrame): dataframe with the turbines that exist in training
            train (bool): if the clustering is being used in training or on new turbines
            *args (pandas.DataFrame): dataframe with new turbines to cluster based on the training turbines

        Returns:
            turb_info (pandas.DataFrame): turbine metadata with assigned cluster column
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
