import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


class ClusteringQuality:
    """
    A class for evaluating clustering quality using multiple metrics.
    This class is intended to work alongside the outputs from your Clustering class.
    """

    def __init__(self, X):
        """
        Initializes the ClusteringQuality class with a dataset.

        Parameters:
        -----------
        X : array-like or pandas DataFrame
            The dataset on which clustering quality will be evaluated.
        """
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        else:
            self.X = np.array(X)

    def silhouette(self, labels):
        """
        Compute the silhouette score.

        Parameters:
        -----------
        labels : array-like
            Cluster labels for each sample.

        Returns:
        --------
        float or None:
            The silhouette score. Returns None if less than 2 clusters are found.
        """
        if len(np.unique(labels)) < 2:
            print("Silhouette score requires at least 2 clusters.")
            return None
        return silhouette_score(self.X, labels)

    def calinski_harabasz(self, labels):
        """
        Compute the Calinski-Harabasz score.

        Parameters:
        -----------
        labels : array-like
            Cluster labels for each sample.

        Returns:
        --------
        float or None:
            The Calinski-Harabasz score. Returns None if less than 2 clusters are found.
        """
        if len(np.unique(labels)) < 2:
            print("Calinski-Harabasz score requires at least 2 clusters.")
            return None
        return calinski_harabasz_score(self.X, labels)

    def davies_bouldin(self, labels):
        """
        Compute the Davies-Bouldin score.

        Parameters:
        -----------
        labels : array-like
            Cluster labels for each sample.

        Returns:
        --------
        float or None:
            The Davies-Bouldin score. Returns None if less than 2 clusters are found.
        """
        if len(np.unique(labels)) < 2:
            print("Davies-Bouldin score requires at least 2 clusters.")
            return None
        return davies_bouldin_score(self.X, labels)

    def evaluate_all(self, labels):
        """
        Evaluate several clustering quality metrics for the given labels.

        Parameters:
        -----------
        labels : array-like
            Cluster labels for each sample.

        Returns:
        --------
        dict:
            A dictionary containing the silhouette score, Calinski-Harabasz score,
            and Davies-Bouldin score.
        """
        results = {
            "silhouette": self.silhouette(labels),
            "calinski_harabasz": self.calinski_harabasz(labels),
            "davies_bouldin": self.davies_bouldin(labels),
        }
        return results

    def evaluate_multiple(self, labels_dict):
        """
        Evaluate clustering quality metrics for multiple clustering results.

        Parameters:
        -----------
        labels_dict : dict
            A dictionary where keys are method names (str) and values are cluster label arrays.

        Returns:
        --------
        dict:
            A dictionary where each key is a method name and its value is another dictionary of quality metrics.
        """
        evaluations = {}
        for method, labels in labels_dict.items():
            evaluations[method] = self.evaluate_all(labels)
        return evaluations
