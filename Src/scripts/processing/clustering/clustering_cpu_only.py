# utils/clustering_cpu_only.py

import pandas as pd

from scripts.processing.clustering.cluster import Clustering


CLUSTERS_CPU = {
    "KMEANS_CPU": {
        "description": "K-Means (CPU)",
        "parameters": [
            {"name": "n_clusters", "type": "int", "default": 3, "min": 2, "max": 50},
            {"name": "random_state", "type": "int", "default": 42},
        ],
    },
    "AGGLOMERATIVE_CPU": {
        "description": "Agglomerative Clustering (CPU)",
        "parameters": [
            {"name": "n_clusters", "type": "int", "default": 3, "min": 2, "max": 50},
            {
                "name": "linkage",
                "type": "list",
                "default": "ward",
                "options": ["ward", "complete", "average", "single"],
            },
        ],
    },
    "DBSCAN_CPU": {
        "description": "DBSCAN (CPU)",
        "parameters": [
            {"name": "eps", "type": "float", "default": 0.5, "min": 0.1, "max": 10.0},
            {"name": "min_samples", "type": "int", "default": 5, "min": 1, "max": 100},
        ],
    },
    "GMM_CPU": {
        "description": "Gaussian Mixture (CPU)",
        "parameters": [
            {"name": "n_components", "type": "int", "default": 3, "min": 2, "max": 50},
            {"name": "random_state", "type": "int", "default": 42},
        ],
    },
}

# "KPROTOTYPES_CPU": {
#   "description": "K-Prototypes (CPU)",
# "parameters": [
#      {"name": "n_clusters", "type": "int", "default": 3, "min": 2, "max": 50},
#     {"name": "init", "type": "str", "default": "Cao"},  # 'Huang', 'Cao', etc.
#      {"name": "n_init", "type": "int", "default": 10, "min": 1, "max": 50},
#     {"name": "verbose", "type": "int", "default": 1, "min": 0, "max": 3}
#  ]
# }


def run_cpu_clustering(method_name: str, data: pd.DataFrame, params: dict) -> pd.Series:
    print(f"Running {method_name}")
    """
    Run the chosen CPU-based clustering on `data` using the provided params.
    Return a pd.Series of cluster labels.

    method_name : str
        One of the keys in CLUSTERS_CPU (e.g. 'KMEANS_CPU', 'DBSCAN_CPU', etc.).
    data : pd.DataFrame
        The dataset (numeric columns) to cluster.
    params : dict
        User-specified parameters (e.g. {'n_clusters': 4, ...}).

    Returns:
        pd.Series: The cluster labels (length = number of rows in 'data').
    """
    # 1) Instantiate the clustering class with the user data
    cluster_model = Clustering(data)

    # 2) Dispatch based on method_name
    if method_name == "KMEANS_CPU":
        model, labels = cluster_model.kmeans(
            n_clusters=params.get("n_clusters", 3),
            random_state=params.get("random_state", 42),
        )
        return pd.Series(labels, name="Cluster")

    elif method_name == "AGGLOMERATIVE_CPU":
        model, labels = cluster_model.agglomerative(
            n_clusters=params.get("n_clusters", 3),
            linkage=params.get("linkage", "ward"),  # Use the first option as default
        )
        return pd.Series(labels, name="Cluster")

    elif method_name == "DBSCAN_CPU":
        model, labels = cluster_model.dbscan(
            eps=params.get("eps", 0.5), min_samples=params.get("min_samples", 5)
        )
        return pd.Series(labels, name="Cluster")

    elif method_name == "GMM_CPU":
        model, labels = cluster_model.gmm(
            n_components=params.get("n_components", 3),
            random_state=params.get("random_state", 42),
        )
        return pd.Series(labels, name="Cluster")

    elif method_name == "KPROTOTYPES_CPU":
        # K-Prototypes can require specifying which columns are categorical
        # In your class, you do something like:
        #   labels = model.fit_predict(self.X, categorical=np.where(self.X.ctypes == 'O')[0])
        # or np.where(self.X.dtypes == 'O')[0]
        model, labels = cluster_model.kprototypes(
            n_clusters=params.get("n_clusters", 3),
            init=params.get("init", "Cao"),
            n_init=params.get("n_init", 10),
            verbose=params.get("verbose", 1),
        )
        return pd.Series(labels, name="Cluster")

    # Fallback if unknown method
    print(f"[run_cpu_clustering] Unrecognized method: {method_name}")
    return pd.Series([-1] * len(data), name="Cluster")
