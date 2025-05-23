# utils/dr_cpu_only.py

import pandas as pd

from scripts.processing.dr.DR import DimensionalityReduction


DR_METHODS_CPU = {
    "PCA": {
        "description": "Principal Component Analysis (CPU)",
        "parameters": [
            {"name": "n_components", "type": "int", "default": 2, "min" : 2, "max" : 3}
        ],
    },
    "FACTOR_ANALYSIS": {
        "description": "Factor Analysis (CPU)",
        "parameters": [
            {"name": "n_components", "type": "int", "default": 2, "min" : 2, "max" : 3}
        ],
    },
    "LLE": {
        "description": "Locally Linear Embedding (CPU)",
        "parameters": [
            {"name": "n_components", "type": "int", "default": 2, "min" : 2, "max" : 3},
            {"name": "n_neighbors", "type": "int", "default": 10, "min": 2, "max": 200},
            {
                "name": "method",
                "type": "list",
                "default": "standard",
                "options": ["standard", "modified"],
            },
        ],
    },
    "UMAP": {
        "description": "UMAP (CPU)",
        "parameters": [
            {"name": "n_components", "type": "int", "default": 2, "min" : 2, "max" : 3},
            {"name": "n_neighbors", "type": "int", "default": 15, "min": 2, "max": 200},
            {
                "name": "min_dist",
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "metric",
                "type": "list",
                "default": "correlation",
                "options": ["euclidean", "manhattan", "cosine", "correlation"],
            },
        ],
    },
    "TSNE": {
        "description": "t-SNE (CPU)",
        "parameters": [
            {"name": "n_components", "type": "int", "default": 2, "min" : 2, "max" : 3},
            {"name": "perplexity", "type": "int", "default": 30, "min": 5, "max": 100},
            {"name": "learning_rate", "type": "int", "default": 200, "min": 10, "max": 1000},
        ],
    },
}


def run_cpu_dr(method_name: str, data: pd.DataFrame, params: dict) -> pd.DataFrame:
    print(f"Running CPU DR method: {method_name}")
    """
    Run the chosen CPU-based DR method on `data` with the given `params`.
    Return a DataFrame with the transformed coordinates (2D or 3D).
    """
    # 1) Instantiate the DR class with the user data
    dr_model = DimensionalityReduction(data)

    # 2) Dispatch to the correct DR method
    if method_name == "PCA":
        coords = dr_model.pca(n_components=params.get("n_components", 2))
        col_names = [f"PC{i+1}" for i in range(coords.shape[1])]
        return pd.DataFrame(coords, columns=col_names)

    elif method_name == "KERNEL_PCA":
        coords = dr_model.kernel_pca(
            n_components=params.get("n_components", 2),
            kernel=params.get("kernel", "linear"),
            gamma=params.get("gamma", None),
            degree=params.get("degree", 3),
        )
        col_names = [f"KPCA{i+1}" for i in range(coords.shape[1])]
        return pd.DataFrame(coords, columns=col_names)

    elif method_name == "FACTOR_ANALYSIS":
        coords = dr_model.factor_analysis(n_components=params.get("n_components", 2))
        col_names = [f"FA{i+1}" for i in range(coords.shape[1])]
        return pd.DataFrame(coords, columns=col_names)

    elif method_name == "LLE":
        coords = dr_model.lle(
            n_components=params.get("n_components", 2),
            n_neighbors=params.get("n_neighbors", 10),
            method=params.get("method", "standard"),
        )
        col_names = [f"LLE{i+1}" for i in range(coords.shape[1])]
        return pd.DataFrame(coords, columns=col_names)

    elif method_name == "UMAP":
        #try:
        print("Initiating UMAP with params:")
        try:
            coords = dr_model._umap(
                n_components=params.get("n_components", 2),
                n_neighbors=params.get("n_neighbors", 15),
                min_dist=params.get("min_dist", 0.1),
                metric=params.get("metric", "euclidean")
            )
            col_names = [f"UMAP{i+1}" for i in range(coords.shape[1])]
        except Exception as e:
            print("Error running UMAP (1):", e)
        print("Done running UMAP")
        return pd.DataFrame(coords, columns=col_names)
    
    elif method_name == "TSNE":
        #try:
        print("Initiating TSNE with params:")
        try:
            coords = dr_model._tsne(
                n_components=params.get("n_components", 2),
                perplexity=params.get("perplexity", 30),
                learning_rate=params.get("learning_rate", 200),
            )
            col_names = [f"TSNE{i+1}" for i in range(coords.shape[1])]
        except Exception as e:
            print("Error running TSNE (1):", e)
        print("Done running TSNE")

        return pd.DataFrame(coords, columns=col_names)

    else:
        print(
            f"[run_cpu_dr] Unrecognized method name: {method_name}. Returning empty DataFrame."
        )
        return pd.DataFrame()
