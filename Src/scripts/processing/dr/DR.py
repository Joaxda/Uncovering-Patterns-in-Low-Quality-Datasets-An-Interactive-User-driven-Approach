from sklearn.manifold import Isomap
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import LocallyLinearEmbedding
from umap import UMAP
from sklearn.manifold import TSNE


class DimensionalityReduction:
    def __init__(self, X):
        """
        Initializes the DR class with a dataset.

        Parameters:
            X : array-like or pandas DataFrame
                The dataset on which to perform dimensionality reduction.
        """
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        else:
            self.X = np.array(X)

    # ----------------------------------------------------
    # Manual methods (user-specified hyperparameters)
    # ----------------------------------------------------
    def pca(self, n_components=2, **kwargs):
        """Principal Component Analysis."""
        model = PCA(n_components=n_components, **kwargs)
        return model.fit_transform(self.X)

    def kernel_pca(
        self, n_components=2, kernel="linear", gamma=None, degree=3, **kwargs
    ):
        """Kernel PCA for non-linear DR."""

        model = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            n_jobs=-1,
            **kwargs
        )
        return model.fit_transform(self.X)

    def factor_analysis(self, n_components=2, **kwargs):
        """Factor Analysis for DR."""
        model = FactorAnalysis(n_components=n_components, **kwargs)
        return model.fit_transform(self.X)

    def isomap(self, n_components=2, n_neighbors=5, **kwargs):
        """Isomap for non-linear DR."""
        model = Isomap(
            n_components=n_components, n_neighbors=n_neighbors, n_jobs=-1, **kwargs
        )
        return model.fit_transform(self.X)

    def lle(self, n_components=2, n_neighbors=10, method="standard", **kwargs):
        """Locally Linear Embedding (LLE)."""
        model = LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=n_components,
            method=method,
            n_jobs=-1,
            **kwargs
        )
        return model.fit_transform(self.X)

    def _umap(self, n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean"):
        """Uniform Manifold Approximation and Projection (UMAP)."""

        print("Running UMAP...")
        try:
            model = UMAP(
                n_components=n_components,
                low_memory=True,
                init="random",
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                n_jobs=-1,
            )
        except Exception as e:
            print("Error in creation of UMAP object:", e)
            return None
        try:
            print("Running UMAP fit...")
            result = model.fit_transform(self.X)
            print("Done with UMAP fit", result)
        except Exception as e:
            print("Error occured in UMAP fit:", e)
            return None
        return result

    def _tsne(self, n_components=2, perplexity=30, learning_rate=200):
        """t-SNE"""

        print("Running TSNE...")
        try:
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                metric="cityblock",
                n_jobs=-1,
            )
        except Exception as e:
            print("Error in creation of TSNE object:", e)
            return None

        try:
            print("Running TSNE fit...")
            result = model.fit_transform(self.X)
            print("Done with TSNE fit", result)
        except Exception as e:
            print("Error occured in TSNE fit:", e)
            return None

        return result
