import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from kmodes.kprototypes import KPrototypes

cpu_cores = os.cpu_count()


class Clustering:
    def __init__(self, X):
        """
        Initializes the Clustering class with a dataset.
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            The dataset on which to perform clustering.
        """
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        else:
            self.X = np.array(X)

    def kmeans(self, n_clusters=3, random_state=42, **kwargs):
        model = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
        labels = model.fit_predict(self.X)
        return model, labels

    def agglomerative(self, n_clusters=3, linkage="ward", **kwargs):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
        labels = model.fit_predict(self.X)
        return model, labels

    def dbscan(self, eps=0.5, min_samples=5, **kwargs):
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=cpu_cores - 1, **kwargs)
        labels = model.fit_predict(self.X)
        return model, labels

    def gmm(self, n_components=3, random_state=42, **kwargs):
        model = GaussianMixture(n_components=n_components, random_state=random_state, **kwargs)
        labels = model.fit_predict(self.X)
        return model, labels

    def spectral(self, n_clusters=3, random_state=42, **kwargs):
        model = SpectralClustering(n_clusters=n_clusters, random_state=random_state, **kwargs)
        labels = model.fit_predict(self.X)
        return model, labels

    def kprototypes(self, n_clusters=3, init="Cao", n_init=10, verbose=0, **kwargs):
        model = KPrototypes(n_clusters=n_clusters, init=init, n_init=n_init, verbose=verbose, **kwargs)
        labels = model.fit_predict(self.X, categorical=[], verbose=verbose)
        return model, labels

    def silhouette(self, labels):
        return silhouette_score(self.X, labels)

    def kmeans_auto(self, cluster_range=None, random_state=42, patience=3, **kwargs):
            """
            Auto-tune KMeans using a range of cluster numbers and early stopping.
            
            Parameters:
            -----------
            patience : int, optional (default=3)
                Number of iterations without improvement before stopping early.
            """
            if cluster_range is None:
                cluster_range = range(2, min(50, len(self.X)) + 1)

            best_score = -np.inf
            best_model, best_labels, best_n = None, None, None
            no_improvement_count = 0

            for n in cluster_range:
                try:
                    model, labels = self.kmeans(n_clusters=n, random_state=random_state, **kwargs)
                    if len(np.unique(labels)) < 2:
                        continue
                    score = self.silhouette(labels)
                    print(f"[KMeans Auto] n_clusters={n}, silhouette score={score}")

                    if score > best_score:
                        best_score = score
                        best_model, best_labels, best_n = model, labels, n
                        no_improvement_count = 0  # Reset counter when improvement occurs
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= patience:
                        print(f"[KMeans Auto] Early stopping triggered at n_clusters={n}")
                        break

                except Exception as e:
                    print(f"[KMeans Auto] Exception for n_clusters={n}: {e}")

            if best_n is None:
                best_n = 3  # Default fallback
            return best_model, best_labels, best_n, best_score

    def dbscan_auto(self, eps_values, min_samples_values, patience=5, **kwargs):
        print(f"[DBSCAN Auto] Using eps_values={eps_values}, min_samples_values={min_samples_values}, patience={patience} for tuning.")
        best_score = -np.inf
        best_model, best_labels, best_params = None, None, None
        no_improvement_count = 0  # Track iterations without improvement

        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    model, labels = self.dbscan(eps=eps, min_samples=min_samples, **kwargs)
                    unique_labels = np.unique(labels)
                    if len(unique_labels) == 1 and unique_labels[0] == -1:
                        print(f"[DBSCAN Auto] No valid clusters found for eps={eps}, min_samples={min_samples}")
                        continue

                    score = self.silhouette(labels)
                    print(f"[DBSCAN Auto] eps={eps}, min_samples={min_samples}, silhouette score={score}")

                    if score > best_score:
                        best_score = score
                        best_model, best_labels = model, labels
                        best_params = {"eps": eps, "min_samples": min_samples}
                        no_improvement_count = 0  # Reset patience counter
                    else:
                        no_improvement_count += 1  # Increment patience counter

                    # Stop if no improvement for the specified number of iterations
                    if no_improvement_count >= patience:
                        print("[DBSCAN Auto] Early stopping due to lack of improvement.")
                        return best_model, best_labels, best_params, best_score

                except Exception as e:
                    print(f"[DBSCAN Auto] Exception for eps={eps}, min_samples={min_samples}: {e}")

        if best_params is None:
            best_params = {"eps": 0.5, "min_samples": 5}
        return best_model, best_labels, best_params, best_score


    def agglomerative_auto(self, cluster_range=None, linkages=None, patience=3, **kwargs):
        """
        Auto-tune Agglomerative Clustering using multiple linkage criteria and early stopping.
        """
        if cluster_range is None:
            cluster_range = range(2, min(50, len(self.X)) + 1)
        if linkages is None:
            linkages = ["ward", "complete", "average", "single"]

        best_score = -np.inf
        best_model, best_labels, best_params = None, None, None
        no_improvement_count = 0

        for linkage in linkages:
            for n in cluster_range:
                try:
                    model, labels = self.agglomerative(n_clusters=n, linkage=linkage, **kwargs)
                    if len(np.unique(labels)) < 2:
                        continue

                    score = self.silhouette(labels)
                    print(f"[Agglomerative Auto] n_clusters={n}, linkage={linkage}, silhouette score={score}")

                    if score > best_score:
                        best_score = score
                        best_model, best_labels = model, labels
                        best_params = {"n_clusters": n, "linkage": linkage}
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= patience:
                        print(f"[Agglomerative Auto] Early stopping triggered at n_clusters={n}, linkage={linkage}")
                        break

                except Exception as e:
                    print(f"[Agglomerative Auto] Exception for n_clusters={n}, linkage={linkage}: {e}")

        if best_params is None:
            best_params = {"n_clusters": 3, "linkage": linkages[0]}
        return best_model, best_labels, best_params, best_score

    def gmm_auto(self, component_range=None, random_state=42, patience=3, **kwargs):
        """
        Auto-tune GMM using a range of component numbers and early stopping.
        """
        if component_range is None:
            component_range = range(2, min(50, len(self.X)) + 1)

        best_score = -np.inf
        best_model, best_labels, best_n = None, None, None
        no_improvement_count = 0

        for n in component_range:
            try:
                model, labels = self.gmm(n_components=n, random_state=random_state, **kwargs)
                if len(np.unique(labels)) < 2:
                    continue

                score = self.silhouette(labels)
                print(f"[GMM Auto] n_components={n}, silhouette score={score}")

                if score > best_score:
                    best_score = score
                    best_model, best_labels, best_n = model, labels, n
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    print(f"[GMM Auto] Early stopping triggered at n_components={n}")
                    break

            except Exception as e:
                print(f"[GMM Auto] Exception for n_components={n}: {e}")

        if best_n is None:
            best_n = 3
        return best_model, best_labels, best_n, best_score
