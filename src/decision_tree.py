"""
File: decision_tree.py
Description: This file contains the implementation of the decision tree algorithm
             for classification purposes. The algorithm will use entropy and information gain
             to build a binary decision tree.
Algorithm: ID3
Author: hLiang
Date: 26/09/2024
Version: 1.0
"""

import math
from collections import Counter
import logging

class DecisionTreeClassifier:
    """
    A Decision Tree Classifier that uses entropy and information gain
    to split the dataset and make decisions.
    """

    def __init__(self, max_depth=None):
        """
        Initialize the decision tree classifier.

        Parameters:
            max_depth (int): The maximum depth of the decision tree. If None, the tree grows until pure.
        """
        self.max_depth = max_depth
        self.tree = None

    def __getstate__(self):
        # 返回要序列化的状态（只保存 max_depth 和 tree）
        return {'max_depth': self.max_depth, 'tree': self.tree}

    def __setstate__(self, state):
        # 反序列化时设置状态
        self.max_depth = state['max_depth']
        self.tree = state['tree']

    def fit(self, X, y):
        """
        Build the decision tree classifier from the training set (X, y).

        Parameters:
            X (list of lists): Training data features.
            y (list): Training data labels.
        """
        data = [x + [label] for x, label in zip(X, y)]
        self.tree = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth):
        """
        Recursively builds the decision tree using the data.

        Parameters:
            data (list of lists): The dataset where the last column is the label.
            depth (int): The current depth of the tree.

        Returns:
            dict: A dictionary representing the tree.
        """
        labels = [row[-1] for row in data]
        if labels.count(labels[0]) == len(labels):  # Pure node (all labels are the same)
            return labels[0]

        if self.max_depth is not None and depth >= self.max_depth:  # Maximum depth reached
            return Counter(labels).most_common(1)[0][0]

        # Find the best split
        best_feature, best_threshold = self._find_best_split(data)
        if best_feature is None:
            return Counter(labels).most_common(1)[0][0]

        left_split, right_split = self._split_data(data, best_feature, best_threshold)
        logging.info(f'Construct: feature - {best_feature} | threshold - {best_threshold}')
        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(left_split, depth + 1),
            "right": self._build_tree(right_split, depth + 1)
        }

    def _find_best_split(self, data):
        """
        Find the best feature and threshold to split the data.

        Parameters:
            data (list of lists): The dataset.

        Returns:
            tuple: The index of the best feature and the best threshold for splitting.
        """
        n_features = len(data[0]) - 1  # Number of features (excluding label)
        best_gain = 0
        best_feature, best_threshold = None, None

        for feature_idx in range(n_features):
            thresholds = set([row[feature_idx] for row in data])
            for threshold in thresholds:
                gain = self._information_gain(data, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature_idx, threshold

        return best_feature, best_threshold

    def _information_gain(self, data, feature_idx, threshold):
        """
        Calculate the information gain for a given feature and threshold.

        Parameters:
            data (list of lists): The dataset.
            feature_idx (int): The index of the feature to split on.
            threshold (any): The threshold value to split on.

        Returns:
            float: The information gain.
        """
        left_split, right_split = self._split_data(data, feature_idx, threshold)
        if not left_split or not right_split:
            return 0

        total_entropy = self._entropy([row[-1] for row in data])
        left_entropy = self._entropy([row[-1] for row in left_split])
        right_entropy = self._entropy([row[-1] for row in right_split])

        p_left = len(left_split) / len(data)
        p_right = 1 - p_left

        return total_entropy - p_left * left_entropy - p_right * right_entropy

    def _entropy(self, labels):
        """
        Calculate the entropy of a label distribution.

        Parameters:
            labels (list): List of labels.

        Returns:
            float: The entropy value.
        """
        counts = Counter(labels)
        total = len(labels)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    def _split_data(self, data, feature_idx, threshold):
        """
        Split the dataset based on a feature and threshold.

        Parameters:
            data (list of lists): The dataset.
            feature_idx (int): The index of the feature to split on.
            threshold (any): The threshold value to split on.

        Returns:
            tuple: Two lists representing the left and right splits.
        """
        left_split = [row for row in data if row[feature_idx] <= threshold]
        right_split = [row for row in data if row[feature_idx] > threshold]
        return left_split, right_split

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
            X (list of lists): Test data features.

        Returns:
            list: Predicted class labels.
        """
        return [self._predict_row(row, self.tree) for row in X]

    def _predict_row(self, row, node):
        """
        Recursively predicts the label for a single data point.

        Parameters:
            row (list): A single data point.
            node (dict or label): The current node of the tree.

        Returns:
            any: The predicted label.
        """
        if isinstance(node, dict):
            if row[node["feature"]] <= node["threshold"]:
                return self._predict_row(row, node["left"])
            else:
                return self._predict_row(row, node["right"])
        else:
            return node
