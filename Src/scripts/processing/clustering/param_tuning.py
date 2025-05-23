import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from scripts.processing.clustering.cluster import Clustering

def get_best_params(technique_name: str, data: pd.DataFrame, **kwargs) -> dict:
    """
    Automatically tune clustering parameters by leveraging the _auto functions in Clustering.
    
    Supported techniques:
      - "KMEANS_CPU": Uses kmeans_auto
      - "AGGLOMERATIVE_CPU": Uses agglomerative_auto
      - "GMM_CPU": Uses gmm_auto
      - "DBSCAN_CPU": Uses dbscan_auto
      
    Optional kwargs can override default candidate ranges:
      - For KMeans and GMM: "cluster_range" or "component_range" (default: range(2, min(50, len(data))+1))
      - For Agglomerative: "cluster_range" and "linkages" (default: ["ward", "complete"])
      - For DBSCAN: "eps_values" (default: np.arange(0.1, 2.1, 0.1)) and "min_samples_values" (default: [5])
    
    Returns:
      dict: A parameters dictionary for the best candidate based on silhouette score.
    """
    cluster_obj = Clustering(data)
    
    if technique_name == "KMEANS_CPU":
        # Use kmeans_auto to scan a range for the best n_clusters.
        cluster_range = kwargs.pop("cluster_range", range(2, min(50, len(data)) + 1))
        _, _, best_n, best_score = cluster_obj.kmeans_auto(cluster_range=cluster_range, random_state=42, **kwargs)
        best_params = {"n_clusters": best_n, "random_state": 42}
        return best_params

    elif technique_name == "AGGLOMERATIVE_CPU":
        # Use agglomerative_auto over a range of cluster numbers as well as linkage methods.
        cluster_range = kwargs.pop("cluster_range", range(2, min(50, len(data)) + 1))
        linkages = kwargs.pop("linkages", ["ward", "complete"])
        _, _, best_params, best_score = cluster_obj.agglomerative_auto(cluster_range=cluster_range, linkages=linkages, **kwargs)
        # best_params already contains keys for "n_clusters" and "linkage"
        return best_params

    elif technique_name == "GMM_CPU":
        # Use gmm_auto to select the best number of components.
        component_range = kwargs.pop("component_range", range(2, min(50, len(data)) + 1))
        _, _, best_n, best_score = cluster_obj.gmm_auto(component_range=component_range, random_state=42, **kwargs)
        best_params = {"n_components": best_n, "random_state": 42}
        return best_params

    elif technique_name == "DBSCAN_CPU":
        # Use dbscan_auto to scan over eps and min_samples.
        eps_values = np.arange(0.1, 5.0, 0.1)
        min_samples_values = [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19,
            20, 25, 30, 40, 50, 75, 100
        ]
        _, _, best_params, best_score = cluster_obj.dbscan_auto(eps_values=eps_values, min_samples_values=min_samples_values, **kwargs)
        return best_params
    
    else:
        raise ValueError(f"Unrecognized technique: {technique_name}")
