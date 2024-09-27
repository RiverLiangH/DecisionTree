"""
File: utils.py
Description: This file contains utility functions for data loading, preprocessing,
             and handling missing values for the Census Income dataset.

Author: hLiang
Date: 26/09/2024
Version: 1.2
"""

import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Remove whitespace and split by comma
    data = [line.strip().split(', ') for line in lines if line.strip() != '']
    return data


def handle_missing_values(data):
    for row in data:
        for i, value in enumerate(row):
            if value == '?':
                # Apply a simple heuristic (e.g., replace with 'Unknown' for categorical data)
                row[i] = 'Unknown'
    return data


def split_features_labels(data):
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    return X, y


def encode_categorical_features(X, mappings, categorical_columns):
    """
    Encode categorical features using mappings.

    Parameters:
    - X: The input feature matrix
    - mappings: A list of dictionaries where each dictionary stores the mapping for a column
    - categorical_columns: List of indices of categorical columns

    Returns:
    - X: The encoded feature matrix
    """
    # Replace each categorical value with its corresponding integer
    for row in X:
        for col_idx in categorical_columns:
            row[col_idx] = mappings[col_idx].get(row[col_idx], row[col_idx])  # Replace categorical

    return X


def create_mappings(X, categorical_columns, numerical_columns, means, std_devs):
    """
    Create mappings for categorical and binned numerical features.

    Parameters:
    - X: The input feature matrix
    - categorical_columns: List of indices of categorical columns
    - numerical_columns: List of indices of numerical columns
    - means: Dictionary containing mean values for each numerical column
    - std_devs: Dictionary containing standard deviation for each numerical column

    Returns:
    - mappings: A list of dictionaries with mappings for each column
    """
    mappings = [{} for _ in range(len(X[0]))]

    # Create mappings for categorical columns
    for col_idx in categorical_columns:
        unique_values = set(row[col_idx] for row in X)
        for i, value in enumerate(unique_values):
            mappings[col_idx][value] = i

    # Create mappings for binned numerical columns
    for col_idx in numerical_columns:
        mean = means[col_idx]
        std_dev = std_devs[col_idx]
        bins = [-np.inf, mean - 2 * std_dev, mean - 1 * std_dev, mean + 1 * std_dev, mean + 2 * std_dev, np.inf]
        for row in X:
            bin_idx = np.digitize(float(row[col_idx]), bins)
            if bin_idx not in mappings[col_idx]:
                mappings[col_idx][bin_idx] = len(mappings[col_idx])

    return mappings


def bin_numerical_features(X, numerical_columns, mappings, means, std_devs):
    """
    Bin numerical features into categories based on mean and standard deviation.

    Parameters:
    - X: The input feature matrix
    - numerical_columns: List of indices of numerical columns
    - mappings: Dictionary of mappings for categorical and binned numerical features
    - means: Dictionary containing mean values for each numerical column
    - std_devs: Dictionary containing standard deviation for each numerical column

    Returns:
    - X: The modified feature matrix with binned numerical features
    """
    for row in X:
        for col_idx in numerical_columns:
            mean = means[col_idx]
            std_dev = std_devs[col_idx]
            bins = [-np.inf, mean - 2 * std_dev, mean - 1 * std_dev, mean + 1 * std_dev, mean + 2 * std_dev, np.inf]
            bin_idx = np.digitize(float(row[col_idx]), bins)
            row[col_idx] = mappings[col_idx][bin_idx]

    return X


def normalize_features(X, numerical_columns):
    """
    Normalize the numerical features in the dataset.

    Parameters:
    - X: The input feature matrix
    - numerical_columns: List of indices of numerical columns

    Returns:
    - X: The normalized feature matrix
    - means: Dictionary containing mean values for each numerical column
    - std_devs: Dictionary containing standard deviation for each numerical column
    """
    means = {}
    std_devs = {}

    # Calculate mean and standard deviation for each numerical column
    for col_idx in numerical_columns:
        col_values = [float(row[col_idx]) for row in X]
        mean = sum(col_values) / len(col_values)
        std_dev = (sum([(x - mean) ** 2 for x in col_values]) / len(col_values)) ** 0.5
        means[col_idx] = mean
        std_devs[col_idx] = std_dev

    return X, means, std_devs


def data_handler(train_data, test_data):
    """
    Preprocess the data including handling missing values, encoding categorical features,
    normalizing numerical features, and binning numerical features.

    Parameters:
    - train_data: The training dataset
    - test_data: The test dataset

    Returns:
    - X_train: Preprocessed training features
    - y_train: Training labels
    - X_test: Preprocessed test features
    - y_test: Test labels
    - means: Dictionary containing mean values for each numerical column
    - std_devs: Dictionary containing standard deviation for each numerical column
    """
    # Handle missing values
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)

    # Split features and labels
    X_train, y_train = split_features_labels(train_data)
    X_test, y_test = split_features_labels(test_data)

    # Define categorical and numerical columns
    categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]  # Categorical feature indices
    numerical_columns = [0, 2, 4, 10, 11, 12]  # Numerical feature indices

    # Normalize numerical features
    X_train, means, std_devs = normalize_features(X_train, numerical_columns)
    X_test, _, _ = normalize_features(X_test, numerical_columns)

    # Create mappings for categorical and binned numerical features
    mappings = create_mappings(X_train, categorical_columns, numerical_columns, means, std_devs)

    # Encode categorical features
    X_train = encode_categorical_features(X_train, mappings, categorical_columns)
    X_test = encode_categorical_features(X_test, mappings, categorical_columns)

    # Bin numerical features into categories and encode them
    X_train = bin_numerical_features(X_train, numerical_columns, mappings, means, std_devs)
    X_test = bin_numerical_features(X_test, numerical_columns, mappings, means, std_devs)

    return X_train, y_train, means, std_devs, X_test, y_test