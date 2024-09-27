"""
File: utils.py
Description: This file contains utility functions for data loading, preprocessing,
             and handling missing values for the Census Income dataset.

Author: hLiang
Date: 26/09/2024
Version: 1.0
"""

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

def encode_categorical_features(X, categorical_columns):
    # Create a mapping for each categorical column
    mappings = [{} for _ in range(len(X[0]))]

    # Apply encoding only to categorical columns
    for col_idx in categorical_columns:
        unique_values = set(row[col_idx] for row in X)
        # Assign an integer to each unique category
        for i, value in enumerate(unique_values):
            mappings[col_idx][value] = i

    # Replace each categorical value with its corresponding integer
    for row in X:
        for col_idx in categorical_columns:
            row[col_idx] = mappings[col_idx].get(row[col_idx], row[col_idx])  # Replace categorical

    return X, mappings


def normalize_features(X):
    num_cols = [0, 2, 4, 10, 11, 12]  # Indices of numerical columns in the dataset
    means = {}
    std_devs = {}

    # Calculate mean and standard deviation for each numerical column
    for col_idx in num_cols:
        col_values = [float(row[col_idx]) for row in X]
        mean = sum(col_values) / len(col_values)
        std_dev = (sum([(x - mean) ** 2 for x in col_values]) / len(col_values)) ** 0.5
        means[col_idx] = mean
        std_devs[col_idx] = std_dev

        # Standardize the column
        for row in X:
            row[col_idx] = (float(row[col_idx]) - mean) / std_dev if std_dev != 0 else 0

    return X, means, std_devs


def data_handler(train_data, test_data):
    # Handle missing values
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)


    X_train, y_train = split_features_labels(train_data)
    X_test, y_test = split_features_labels(test_data)

    # Encode categorical features in train and test sets
    categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]  # Corresponding to native-country, marital-status, occupation, etc.
    X_train, mappings = encode_categorical_features(X_train, categorical_columns)
    X_test, _ = encode_categorical_features(X_test, categorical_columns)

    # Normalize train and test sets
    X_train, means, std_devs = normalize_features(X_train)
    X_test, _, _ = normalize_features(X_test)

    return X_train, y_train, means, std_devs, X_test, y_test