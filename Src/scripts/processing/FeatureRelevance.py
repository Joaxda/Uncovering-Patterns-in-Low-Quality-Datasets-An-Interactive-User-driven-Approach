import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from dowhy import CausalModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor


class FeatureRelevance:
    def __init__(self, dataset, targets=None):
        self.dataset = dataset
        self.features = dataset.columns
        self.targets = targets if targets else [dataset.columns[-1]]

    def linear_regression(self, stats_model=False):
        X = self.dataset[self.features.difference(self.targets)]
        y = self.dataset[self.targets]
        reg = LinearRegression().fit(X, y)
        print("The model coefficients:", reg.coef_)
        print("The model intercepts:", reg.intercept_)

        if stats_model and len(self.targets) == 1:
            X_const = sm.add_constant(X)
            ols_model_linear = sm.OLS(y, X_const).fit()
            print(ols_model_linear.summary())

    def polynomial_regression(self, deg=2, bias=False, plot=False):
        X = self.dataset[self.features.difference(self.targets)]
        y = self.dataset[self.targets]

        poly = PolynomialFeatures(degree=deg, include_bias=bias)
        X_poly = poly.fit_transform(X)

        model = LinearRegression().fit(X_poly, y)
        feature_names = poly.get_feature_names_out(X.columns)

        # Create a DataFrame for multiple target variables
        coefficients = pd.DataFrame(
            model.coef_.T, index=feature_names, columns=self.targets
        )

        print("Polynomial Regression Coefficients:")
        print(coefficients)

        if plot:
            fig_height = max(8, len(coefficients) * 0.35)
            fig, axes = plt.subplots(nrows=len(self.targets), figsize=(12, fig_height))

            if len(self.targets) == 1:
                axes = [axes]

            for idx, target in enumerate(self.targets):
                sorted_coefs = coefficients[target].abs().sort_values(ascending=False)
                axes[idx].set_ylim(-1, len(sorted_coefs))
                axes[idx].barh(sorted_coefs.index, sorted_coefs.values)
                axes[idx].set_title(
                    f"Polynomial Regression Feature Importance ({target})"
                )
                axes[idx].set_xlabel("Coefficient Value")
                axes[idx].invert_yaxis()
                axes[idx].set_xscale("log")  # Set x-axis to log scale

            plt.tight_layout()
            plt.show()

    def ridge_lasso_regression(self, alpha_ridge=1.0, alpha_lasso=0.01, plot=False):
        X = self.dataset[self.features.difference(self.targets)]
        y = self.dataset[self.targets]

        ridge = Ridge(alpha=alpha_ridge).fit(X, y)
        lasso = Lasso(alpha=alpha_lasso, max_iter=5000).fit(X, y)

        # Handle case where only one target variable exists
        if len(self.targets) == 1:
            ridge_coef_df = pd.DataFrame(
                ridge.coef_.reshape(1, -1), columns=X.columns, index=self.targets
            )
            lasso_coef_df = pd.DataFrame(
                lasso.coef_.reshape(1, -1), columns=X.columns, index=self.targets
            )
        else:
            ridge_coef_df = pd.DataFrame(
                ridge.coef_, columns=X.columns, index=self.targets
            )
            lasso_coef_df = pd.DataFrame(
                lasso.coef_, columns=X.columns, index=self.targets
            )

        print("Ridge Coefficients:")
        print(ridge_coef_df)
        print("\nLasso Coefficients:")
        print(lasso_coef_df)

        if plot:
            fig_height = max(8, len(X.columns) * 0.35)

            if len(self.targets) == 1:
                # Convert to DataFrame for plotting
                ridge_coef = pd.DataFrame(
                    {"Feature": X.columns, "Ridge Coefficient": ridge.coef_.flatten()}
                )
                lasso_coef = pd.DataFrame(
                    {"Feature": X.columns, "Lasso Coefficient": lasso.coef_.flatten()}
                )
            else:
                # Take the first target variable's coefficients
                ridge_coef = pd.DataFrame(
                    {"Feature": X.columns, "Ridge Coefficient": ridge.coef_[0]}
                )
                lasso_coef = pd.DataFrame(
                    {"Feature": X.columns, "Lasso Coefficient": lasso.coef_[0]}
                )

            # Sort for visualization
            ridge_coef = ridge_coef.reindex(
                ridge_coef["Ridge Coefficient"].abs().sort_values(ascending=False).index
            )
            lasso_coef = lasso_coef.reindex(
                lasso_coef["Lasso Coefficient"].abs().sort_values(ascending=False).index
            )

            # Plot Ridge Regression Coefficients
            plt.figure(figsize=(12, fig_height))
            plt.barh(
                ridge_coef["Feature"], ridge_coef["Ridge Coefficient"], color="blue"
            )
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature")
            plt.title("Ridge Regression Feature Importance")
            plt.gca().invert_yaxis()
            plt.subplots_adjust(left=0.3)
            plt.xscale("log")
            plt.show()

            # Plot Lasso Regression Coefficients
            plt.figure(figsize=(12, fig_height))
            plt.barh(
                lasso_coef["Feature"], lasso_coef["Lasso Coefficient"], color="red"
            )
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature")
            plt.title("Lasso Regression Feature Importance")
            plt.gca().invert_yaxis()
            plt.subplots_adjust(left=0.3)
            plt.xscale("log")
            plt.show()

    def poisson_regression(self, plot=False):
        X = self.dataset[self.features.difference(self.targets)]
        y = self.dataset[self.targets[0]]  # Poisson regression is usually single-output
        X_const = sm.add_constant(X)
        poisson_model = sm.GLM(y, X_const, family=sm.families.Poisson()).fit()
        poisson_coef = pd.DataFrame(
            {"Feature": X_const.columns, "Coefficient": poisson_model.params}
        )
        # Sort coefficients by absolute value for better visualization
        poisson_coef = poisson_coef.reindex(
            poisson_coef["Coefficient"].abs().sort_values(ascending=False).index
        )
        print(poisson_coef)
        print(poisson_model.summary())
        if plot:
            # Define dynamic figure height based on number of features
            fig_height = max(8, len(X_const.columns) * 0.35)

            # Plot Poisson Regression Coefficients
            plt.figure(figsize=(12, fig_height))
            plt.barh(
                poisson_coef["Feature"], poisson_coef["Coefficient"], color="green"
            )
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature")
            plt.title("Poisson Regression Feature Importance")
            plt.gca().invert_yaxis()  # Highest impact at the top
            plt.subplots_adjust(left=0.3)  # Adjust left margin for readability
            plt.xscale("log")  # Log scale for better visualization
            plt.show()

    def gradient_boosting(self, esti=100, plot=False):
        X = self.dataset[self.features.difference(self.targets)]
        y = self.dataset[self.targets]
        xgb = XGBRegressor(objective="reg:squarederror", n_estimators=esti).fit(X, y)
        # Create a DataFrame to show feature names and their importance
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": xgb.feature_importances_}
        )
        # Sort by importance (highest first)
        feature_importance = feature_importance.sort_values(
            by="Importance", ascending=True
        )
        # Display the feature importance
        print("\nGradient Boosting Feature Importance:")
        print(feature_importance)
        if plot:
            # Define dynamic figure height based on number of features
            fig_height = max(8, len(X.columns) * 0.35)

            # Plot Feature Importance for XGBoost
            plt.figure(figsize=(12, fig_height))
            plt.barh(
                feature_importance["Feature"],
                feature_importance["Importance"],
                color="purple",
            )
            plt.xlabel("Feature Importance Score")
            plt.ylabel("Feature")
            plt.title("XGBoost Feature Importance")
            plt.gca().invert_yaxis()  # Highest importance at the top
            plt.subplots_adjust(left=0.3)  # Adjust left margin for readability
            plt.xscale("log")  # Log scale for better visualization
            plt.show()

        return feature_importance

    def random_forest_regression(self, esti=100, plot=False):
        X = self.dataset[self.features.difference(self.targets)]
        y = self.dataset[self.targets]
        rf = RandomForestRegressor(n_estimators=esti).fit(X, y)
        # Create a DataFrame to show feature names and their importance
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": rf.feature_importances_}
        )
        # Sort by importance (highest first)
        feature_importance = feature_importance.sort_values(
            by="Importance", ascending=False
        )
        # Display the feature importance
        print(feature_importance)
        if plot:
            # Define dynamic figure height based on number of features
            fig_height = max(8, len(X.columns) * 0.35)

            # Plot Feature Importance for Random Forest
            plt.figure(figsize=(12, fig_height))
            plt.barh(
                feature_importance["Feature"],
                feature_importance["Importance"],
                color="orange",
            )
            plt.xlabel("Feature Importance Score")
            plt.ylabel("Feature")
            plt.title("Random Forest Feature Importance")
            plt.xscale("log")  # Log scale for better visualization
            plt.gca().invert_yaxis()

    def mutual_information(self, plot=False):
        # Select features excluding target(s)
        X = self.dataset[self.features.difference(self.targets)]
        y_df = self.dataset[self.targets]  # Multi-output case

        mi_results = {}

        # Calculate MI scores for each target
        for target in self.targets:
            y = y_df[target].values.ravel()  # Convert to 1D array
            mi_scores = mutual_info_regression(X, y)  # Compute MI for this target
            mi_results[target] = mi_scores  # Store results

        # Create a DataFrame with MI scores for all targets
        mi_df = pd.DataFrame(mi_results, index=X.columns)

        # Print sorted MI scores for each target
        print("\nMutual Information Scores:")
        for target in self.targets:
            print(f"\nTarget: {target}")
            sorted_mi = mi_df[target].sort_values(ascending=True)
            print(sorted_mi)

        if plot:
            fig, axes = plt.subplots(
                nrows=len(self.targets), figsize=(12, 5 * len(self.targets))
            )

            if len(self.targets) == 1:
                axes = [axes]  # Ensure it's iterable for a single target

            for idx, target in enumerate(self.targets):
                sorted_mi = mi_df[target].sort_values(ascending=False)
                axes[idx].barh(sorted_mi.index, sorted_mi.values, color="darkblue")
                axes[idx].set_xlabel("Mutual Information Score")
                axes[idx].set_ylabel("Feature")
                axes[idx].set_title(f"Mutual Information for Target: {target}")
                axes[idx].invert_yaxis()  # Ensures highest scores appear at the top

            plt.tight_layout()
            plt.show()

        return mi_df

    def correlation_analysis(self, method="pearson", plot=False):
        corr_matrix = self.dataset.corr(method=method)
        # Print the names of the columns and index of the correlation matrix
        print("Correlation Matrix Columns:", corr_matrix.columns.tolist())
        print("Correlation Matrix Index:", corr_matrix.index.tolist())
        # we remove the target from the correlation matrix and print the correlation matrix
        print(f"target: {self.targets}")

        target_corr = corr_matrix[self.targets]
        # we remove the target from the target_corr and print the correlation matrix
        target_corr = target_corr.drop(labels=self.targets, axis=0)
        print("Target Correlations:", target_corr)

        if plot:
            plt.figure(figsize=(10, len(self.features) * 0.4))
            sns.heatmap(
                target_corr,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                linewidths=0.5,
            )
            plt.title(f"{method.capitalize()} Correlation with Target Variables")
            plt.show()
        return corr_matrix, target_corr


    def causal_analysis(self):
        X = self.dataset[self.features.difference(self.targets)]
        y = self.dataset[self.targets[0]]  # Requires a single target variable

        # Define the causal model
        causal_model = CausalModel(
            data=self.dataset, treatment=X.columns.tolist(), outcome=self.targets[0]
        )

        identified_estimand = causal_model.identify_effect()
        causal_estimate = causal_model.estimate_effect(
            identified_estimand, method_name="backdoor.linear_regression"
        )

        print("\nEstimated Causal Effects:")
        print(causal_estimate)

        return causal_estimate

    def __optimal_clusters(self, max_clusters=10, method="silhouette"):
        X = self.dataset[self.features.difference(self.targets)].values
        scores = []

        # Try different numbers of clusters
        for k in range(2, max_clusters + 1):
            print(f"Trying {k} clusters...")
            kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
            if method == "silhouette":
                score = silhouette_score(X, kmeans.labels_)
            else:  # Elbow method
                score = kmeans.inertia_
            scores.append(score)

        # Determine the best cluster number
        best_k = (
            scores.index(max(scores)) + 2
            if method == "silhouette"
            else np.argmin(np.diff(scores)) + 2
        )
        # we print the best k
        print(f"Best number of clusters: {best_k}")
        return best_k

    def cluster_features(self, n_clusters=None):
        X = self.dataset[self.features.difference(self.targets)]

        # If cluster number not provided, determine the best one
        if n_clusters is None:
            n_clusters = self.__optimal_clusters()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        self.dataset["Cluster"] = kmeans.labels_
        print(f"Added 'Cluster' column with {n_clusters} clusters.")

        # PCA for visualization (if features > 2)
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            # Get component names based on highest loadings
            loadings = pd.DataFrame(
                pca.components_.T, index=X.columns, columns=["PC1", "PC2"]
            )
            pc1_top_feature = loadings["PC1"].abs().idxmax()
            pc2_top_feature = loadings["PC2"].abs().idxmax()

            plt.figure(figsize=(8, 6))
            plt.scatter(
                X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap="viridis", alpha=0.7
            )
            plt.xlabel(f"Principal Component 1 ({pc1_top_feature})")
            plt.ylabel(f"Principal Component 2 ({pc2_top_feature})")
            plt.title("Clusters Visualized using PCA")
            plt.colorbar(label="Cluster")
            plt.show()

            print("\nPrincipal Component Loadings:")
            print(loadings)

    def recursive_feature_elimination(self, model=None, max_features=10, plot=False):
        X = self.dataset[self.features.difference(self.targets)]
        y = self.dataset[self.targets]

        # Use a stronger model if none is provided
        if model is None:
            model = XGBRegressor(
                objective="reg:squarederror", n_estimators=100
            )  # Handles non-linearity well

        selector = RFE(model, n_features_to_select=max_features)
        selector.fit(X, y)

        selected_features = X.columns[selector.support_]

        print("Selected Features:", list(selected_features))

        # Visualization
        if plot:
            importance = selector.ranking_
            feature_importance = pd.DataFrame(
                {"Feature": X.columns, "Importance": importance}
            )
            feature_importance = feature_importance.sort_values(
                by="Importance", ascending=True
            )

            plt.figure(figsize=(10, 5))
            plt.barh(
                feature_importance["Feature"],
                feature_importance["Importance"],
                color="darkred",
            )
            plt.xlabel("Feature Importance Ranking (Lower is Better)")
            plt.ylabel("Feature")
            plt.title("Recursive Feature Elimination (RFE) Ranking using XGBoost")
            plt.gca().invert_yaxis()
            plt.show()

        return selected_features
