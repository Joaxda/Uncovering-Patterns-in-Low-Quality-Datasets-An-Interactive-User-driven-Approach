import matplotlib
matplotlib.use("TkAgg")

import shap
shap.initjs()

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text

class ClusterExplanation:
    def __init__(self, data, cluster_labels):
        """
        Parameters:
          data: A pandas DataFrame containing your features.
          cluster_labels: An array-like of cluster assignments (one per row in data).
        """
        self.data = data.copy()
        self.data["Cluster"] = (
            cluster_labels  # add cluster labels to the data for grouping
        )
        self.cluster_labels = cluster_labels
        self.tree_model = None

    def train_surrogate_tree(self, max_depth=3):
        """
        Trains a simple decision tree (the surrogate model) to predict the given cluster labels.
        A shallow tree (low max_depth) produces a small number of rules that are easier to understand.

        Parameters:
          max_depth: The maximum depth of the decision tree. A lower value yields fewer rules.
        """
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(self.data.drop(columns=["Cluster"]), self.data["Cluster"])
        self.tree_model = dt
        return dt

    def get_decision_rules(self):
        """
        Extracts decision rules from the trained surrogate decision tree.

        Returns:
          A string containing the human-readable rules.
        """
        if self.tree_model is None:
            self.train_surrogate_tree()
        rules = export_text(
            self.tree_model,
            feature_names=list(self.data.drop(columns=["Cluster"]).columns),
        )
        return rules

    def generate_plain_language_summary(self):
        """
        Prints a plain language summary of the rules that explain how the clusters are separated.
        These rules are converted into everyday language.
        """
        rules = self.get_decision_rules()
        print("Plain Language Explanation of the Clusters (Surrogate Model Rules):\n")
        for line in rules.splitlines():
            # Replace technical operators with plain language equivalents.
            plain_line = (
                line.replace("<= ", "is less than or equal to ")
                .replace("> ", "is greater than ")
                .replace("class: ", "-> then assign to cluster ")
                .replace("|---", "  • ")
            )
            print(plain_line)
        print("\n")  # extra spacing

    def plot_tree(self):
        """
        Plots the surrogate decision tree for a visual representation of the rules.
        """
        if self.tree_model is None:
            self.train_surrogate_tree()
        plt.figure(figsize=(12, 6))
        tree.plot_tree(
            self.tree_model,
            feature_names=self.data.drop(columns=["Cluster"]).columns,
            class_names=[str(c) for c in sorted(set(self.cluster_labels))],
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.title("Surrogate Decision Tree Explaining the Clusters")
        plt.show()

    def generate_detailed_cluster_profile(self):
        """
        Generates a thorough, plain language profile for each cluster.
        It includes:
          - The number of data points in the cluster.
          - The average values for each feature.
          - A comparison between the cluster's averages and the overall dataset's averages.

        This narrative summary can help a non-technical audience understand what makes each cluster unique.
        """
        # Remove the cluster column for feature-wise computations
        features = self.data.drop(columns=["Cluster"]).columns

        overall_means = self.data[features].mean()
        cluster_group = self.data.groupby("Cluster")
        cluster_counts = cluster_group.size()
        cluster_means = cluster_group.mean()

        report_lines = []
        report_lines.append("Detailed Cluster Profiles:\n")
        report_lines.append(
            "Overall, there are {} data points across {} clusters.\n".format(
                len(self.data), len(cluster_counts)
            )
        )

        for cluster, count in cluster_counts.items():
            report_lines.append("Cluster {}: ({} data points)".format(cluster, count))
            cluster_avg = cluster_means.loc[cluster]
            report_lines.append("  • Key average feature values:")
            for feat in features:
                overall_val = overall_means[feat]
                cluster_val = cluster_avg[feat]
                diff = cluster_val - overall_val
                # Create a plain language comparison
                if abs(diff) < 1e-6:
                    comp_text = "about the same as"
                elif diff > 0:
                    comp_text = "higher than"
                else:
                    comp_text = "lower than"
                report_lines.append(
                    "      - {}: {:.2f} ({} overall average of {:.2f})".format(
                        feat, cluster_val, comp_text, overall_val
                    )
                )
            report_lines.append("")  # blank line for separation

        # Print out the detailed report
        detailed_report = "\n".join(report_lines)
        print(detailed_report)
        return detailed_report

    def plot_feature_boxplots(self, features=None):
        """
        Plots box plots for the selected features across clusters.
        This visual comparison helps illustrate the spread and differences in each feature.

        Parameters:
          features: list of feature names to plot. If None, plots all features (except the cluster column).
        """
        if features is None:
            features = self.data.drop(columns=["Cluster"]).columns

        for feature in features:
            if feature not in self.data.columns:
                print(
                    f"Warning: The feature '{feature}' was not found in the data. Skipping it."
                )
                continue
            plt.figure(figsize=(8, 4))
            self.data.boxplot(column=feature, by="Cluster", grid=False)
            plt.title("Distribution of '{}' across Clusters".format(feature))
            plt.suptitle("")  # Remove the automatic 'Boxplot grouped by cluster' title.
            plt.xlabel("Cluster")
            plt.ylabel(feature)
            plt.show()


