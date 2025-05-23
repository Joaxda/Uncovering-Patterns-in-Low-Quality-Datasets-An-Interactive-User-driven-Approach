import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


class DimensionalityReductionQuality:
    """
    A class for evaluating the quality of low-dimensional embeddings (results of DR)
    by comparing the embedding to the original high-dimensional data.
    """

    def __init__(self, X):
        """
        Initializes the quality evaluator with the original high-dimensional data.

        Parameters:
        -----------
        X : array-like or pandas DataFrame
            The original data.
        """
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        else:
            self.X = np.array(X)

    def trustworthiness(self, embedding, n_neighbors=5):
        """
        Compute the trustworthiness of the embedding.

        Parameters:
        -----------
        embedding : array-like, shape (n_samples, n_components)
            The low-dimensional embedding.
        n_neighbors : int, optional (default=5)
            Number of neighbors to consider.

        Returns:
        --------
        float:
            Trustworthiness score (1.0 is best).
        """
        try:
            return trustworthiness(self.X, embedding, n_neighbors=n_neighbors)
        except Exception as e:
            print("Error computing trustworthiness:", e)
            return None

    def continuity(self, embedding, n_neighbors=5):
        """
        Compute the continuity of the embedding.

        Continuity measures how well the high-dimensional
        neighbors of each point are preserved in the low-dimensional embedding.

        Parameters:
        -----------
        embedding : array-like, shape (n_samples, n_components)
            The low-dimensional embedding.
        n_neighbors : int, optional (default=5)
            Number of neighbors to consider.

        Returns:
        --------
        float:
            Continuity score (1.0 is best).
        """
        n = self.X.shape[0]
        if n <= n_neighbors:
            print("Not enough samples for continuity computation.")
            return None

        # Compute pairwise distance matrices
        D_orig = squareform(pdist(self.X))
        D_emb = squareform(pdist(embedding))
        total = 0.0

        for i in range(n):
            # Exclude self from ordering.
            orig_order = np.argsort(D_orig[i])
            emb_order = np.argsort(D_emb[i])
            orig_order = orig_order[orig_order != i]
            emb_order = emb_order[emb_order != i]

            # True neighbors in the original space:
            true_neighbors = set(orig_order[:n_neighbors])
            # Neighbors selected in the embedding:
            estimated_neighbors = set(emb_order[:n_neighbors])
            # Those neighbors that are missing in the embedding:
            missing = true_neighbors - estimated_neighbors

            for j in missing:
                # Get rank of j in the embedding ordering (starting at 1)
                rank = np.where(emb_order == j)[0][0] + 1
                total += rank - n_neighbors

        # Denomintor as in the trustworthiness formulation.
        denom = n * n_neighbors * (2 * n - 3 * n_neighbors - 1)
        continuity_score = 1 - (2.0 / denom) * total
        return continuity_score

    def stress(self, embedding):
        """
        Compute the normalized stress of the embedding.

        Stress is defined as:

            stress = sqrt( sum_{i<j} (d_emb(i,j) - d_orig(i,j))^2 / sum_{i<j} (d_orig(i,j))^2 )

        where d_orig and d_emb are pairwise distances in the original and
        embedded spaces respectively.

        Parameters:
        -----------
        embedding : array-like, shape (n_samples, n_components)
            The low-dimensional embedding.

        Returns:
        --------
        float:
            The stress value (lower is better).
        """
        try:
            orig_dists = pdist(self.X)
            emb_dists = pdist(embedding)
            num = np.sum((emb_dists - orig_dists) ** 2)
            denom = np.sum(orig_dists**2)
            stress_value = np.sqrt(num / denom)
            return stress_value
        except Exception as e:
            print("Error computing stress:", e)
            return None

    def distance_correlation(self, embedding):
        """
        Compute the Pearson correlation coefficient between the pairwise
        distances of the original data and the embedding.

        A higher correlation indicates that global structure is well preserved.

        Parameters:
        -----------
        embedding : array-like, shape (n_samples, n_components)
            The low-dimensional embedding.

        Returns:
        --------
        float:
            The Pearson correlation coefficient.
        """
        try:
            orig_dists = pdist(self.X)
            emb_dists = pdist(embedding)
            corr, _ = pearsonr(orig_dists, emb_dists)
            return corr
        except Exception as e:
            print("Error computing distance correlation:", e)
            return None

    def evaluate_all(self, embedding, n_neighbors=5):
        """
        Evaluate several dimensionality reduction quality metrics for a given embedding.

        Parameters:
        -----------
        embedding : array-like, shape (n_samples, n_components)
            The low-dimensional embedding.
        n_neighbors : int, optional (default=5)
            Number of neighbors to use in neighborhood-based metrics.

        Returns:
        --------
        dict:
            A dictionary containing trustworthiness, continuity, stress, and
            distance correlation scores.
        """
        results = {
            "trustworthiness": self.trustworthiness(embedding, n_neighbors=n_neighbors),
            "continuity": self.continuity(embedding, n_neighbors=n_neighbors),
            "stress": self.stress(embedding),
            "distance_correlation": self.distance_correlation(embedding),
        }
        return results
