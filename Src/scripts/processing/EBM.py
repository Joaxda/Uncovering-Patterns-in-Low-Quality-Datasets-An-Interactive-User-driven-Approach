import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class EBM:
    def __init__(self, X_cluster, y_cluster, CLUSTER_LABEL_NO):
        # X_cluster: Pandas DataFrame or 2D numpy array containing all feature columns (exclude the cluster column).
        self.X_cluster = X_cluster

        # y_cluster: Array-like structure (list, Series, or numpy array) containing cluster labels.
        #            This should include samples from multiple clusters.
        self.y_cluster = y_cluster

        # CLUSTER_LABEL_NO: The specific cluster label you wish to explain.
        self.CLUSTER_LABEL_NO = CLUSTER_LABEL_NO

        # Initialize the Explainable Boosting Classifier with fixed parameters.
        self.ebm = ExplainableBoostingClassifier(
            random_state=42, early_stopping_rounds=100, max_rounds=5000, n_jobs=-1
        )

        # feature_names: To be populated with names of the features from X_cluster.
        self.feature_names = None

        # term_importance: To store the importance values for each term after training.
        self.term_importance = None

        # accuracy: To store the model accuracy on the test set.
        self.accuracy = None

        # importances_df: To store a DataFrame mapping feature (or term) names to their importance scores.
        self.importances_df = None

    def train(self):
        # Convert y_cluster to a binary target:
        # 1 if the sample is from the desired cluster (CLUSTER_LABEL_NO), 0 otherwise.
        y_binary = (self.y_cluster == self.CLUSTER_LABEL_NO).astype(int)

        # Split the data into training and testing sets (40% for testing)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_cluster, y_binary, test_size=0.1, random_state=42
        )

        # If X_cluster is a DataFrame, extract column names for later use.
        self.feature_names = list(X_train.columns)

        # Define feature types for the EBM. Here, all features are auto-detected.
        feature_types = ["auto" for _ in range(len(self.feature_names))]
        self.ebm.feature_types = feature_types

        print("Shape of training data:", X_train.shape)
        print("Shape of training target:", y_train.shape)

        # Train the EBM model on the training data
        self.ebm.fit(X_train, y_train)

        # Predict on the test set and compute accuracy
        y_pred = self.ebm.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print("Model accuracy:", self.accuracy)

        # Retrieve the term importances from the trained model.
        # These importances reflect the contribution of each term in predicting whether
        # a sample belongs to the desired cluster (1) or not (0).
        self.term_importance = self.ebm.term_importances()
        print("Term importances:", self.term_importance)

    def sort_feature_importance(self):
        # Retrieve term features, which are lists of feature indices corresponding to each term.
        term_features = self.ebm.term_features_
        term_names = []
        for feature_indices in term_features:
            # Create a descriptive name for each term by joining the names of its constituent features.
            names = [self.feature_names[i] for i in feature_indices]
            term_name = " & ".join(names)
            term_names.append(term_name)

        # Create a DataFrame mapping each term to its importance.
        self.importances_df = pd.DataFrame(
            {"Feature": term_names, "Importance": self.term_importance}
        )

        # Sort the DataFrame by importance (ascending order)
        self.importances_df = self.importances_df.sort_values(
            by="Importance", ascending=True
        ).reset_index(drop=True)

    def plot_EBM_result(self):
        # Generate the sorted feature importance DataFrame.
        self.sort_feature_importance()
        plt.figure(figsize=(10, 8))
        plt.barh(self.importances_df["Feature"], self.importances_df["Importance"])
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title(f"EBM Feature Importances for Cluster Label {self.CLUSTER_LABEL_NO}")
        plt.tight_layout()
        plt.show()

    def main(self):
        # Train the model using the binary target (desired cluster vs. rest)
        self.train()

        # Sort and prepare the feature importance DataFrame
        self.sort_feature_importance()

        # Optionally, you can plot the feature importances.
        # self.plot_EBM_result()

        # Return the model's accuracy and the DataFrame of feature importances.
        return self.accuracy, self.importances_df
