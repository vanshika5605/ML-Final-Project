# How to run the code?
# Refer README.md

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from load_dataset import load_dataset, load_from_csv, load_student_data_from_csv 

def calculate_metrics(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_metrics = {
        label: {'TP': 0, 'FP': 0, 'FN': 0}
        for label in labels
    }

    for true, pred in zip(y_true, y_pred):
        for label in labels:
            if pred == label and true == label:
                label_metrics[label]['TP'] += 1
            elif pred == label and true != label:
                label_metrics[label]['FP'] += 1
            elif pred != label and true == label:
                label_metrics[label]['FN'] += 1

    precisions, recalls, f1s = [], [], []

    for label in labels:
        TP = label_metrics[label]['TP']
        FP = label_metrics[label]['FP']
        FN = label_metrics[label]['FN']

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return {
        'Accuracy': accuracy,
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1': np.mean(f1s)
    }

def stratified_k_fold(X, y, target_column, k):
    """
    Performs stratified k-fold split on dataset such that each fold maintains 
    the proportion of classes (labels) as in the original dataset.
    """
    Xy = pd.DataFrame(X)
    Xy[target_column] = y

    # Group by label
    label_groups = defaultdict(list)
    for label in np.unique(y):
        label_groups[label] = Xy[Xy[target_column] == label]
    
    # Initialize k empty folds

    # Shuffle and divide each class group into k parts
    folds = [pd.DataFrame() for _ in range(k)]
    for label, group in label_groups.items():
        group = group.sample(frac=1, random_state=None).reset_index(drop=True) # Shuffle group
        parts = np.array_split(group, k) # Split group into k parts

        # Distribute parts into folds
        for i in range(k):
            folds[i] = pd.concat([folds[i], parts[i]], ignore_index=True)

    return folds  # list of k folds, each a DataFrame with X and label

def preprocess_data(train_df, test_df, categorical_cols, numeric_cols, target_column):
    """
    Handles cases where categorical or numerical columns may be empty.
    Returns X_train, X_test, y_train, y_test.
    """
    # Initialize empty arrays in case categories or numerics are missing
    X_cat_train = np.empty((len(train_df), 0))
    X_cat_test = np.empty((len(test_df), 0))

    if categorical_cols:  # Only encode if not empty
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_train = enc.fit_transform(train_df[categorical_cols])
        X_cat_test  = enc.transform(test_df[categorical_cols])

    X_num_train = np.empty((len(train_df), 0))
    X_num_test = np.empty((len(test_df), 0))

    if numeric_cols:
        X_num_train = train_df[numeric_cols].values.astype(float)
        X_num_test  = test_df[numeric_cols].values.astype(float)

        X_num_train_scaled = normalizeFeatures(X_num_train)
        X_min, X_max = X_num_train.min(axis=0), X_num_train.max(axis=0)
        X_num_test_scaled = (X_num_test - X_min) / (X_max - X_min + 1e-8)

        X_num_train = X_num_train_scaled
        X_num_test = X_num_test_scaled

    # Concatenate features (handle cases with only numeric or only categorical)
    X_train = np.hstack([X_num_train, X_cat_train])
    X_test  = np.hstack([X_num_test,  X_cat_test])

    y_train = train_df[target_column].values.reshape(-1, 1)
    y_test  = test_df[target_column].values.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def normalizeFeatures(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denominator = X_max - X_min
    denominator[denominator == 0] = 1e-8  # avoid division by zero
    X_normalized = (X - X_min) / denominator
    return pd.DataFrame(X_normalized)

# Function to calculate the euclidean distance between 2 points
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Function for the KNN Classifier
def knn(X_train, y_train, X_test, k):
    # X_train = X_train.to_numpy()
    # y_train = y_train.to_numpy().flatten()
    # X_test = X_test.to_numpy()
    y_pred = []
    X_test = np.atleast_2d(X_test)

    for x in X_test:
        # Calculating distance of a point x from all points in X_train
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        # Get k nearest neighbors
        sorted_indices = np.argsort(distances)[:k]
        # Get labels of neighbors
        nearest_labels = y_train[sorted_indices]
        # Count occurrences
        unique, counts = np.unique(nearest_labels, return_counts=True)
        # Majority vote
        y_pred.append(unique[np.argmax(counts)])
    return np.array(y_pred)

# Function to create the graph of accuracy vs k
def createGraph(k_values, accuracies, title, ylabel, label):
    stds = np.std(np.array(accuracies), axis=0)

    plt.figure(figsize=(8, 6))
    plt.errorbar(k_values, accuracies, yerr=stds, fmt='-o', capsize=5, label=label)

    plt.xlabel('Value of k')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Part 1 - With Normalization

def run(X, y, categorical_cols, numeric_cols, target_column, num_folds=10):
    k_values = [1, 5, 10, 15, 20, 25, 30, 35]

    print("Creating folds")
    folds = stratified_k_fold(X, y, target_column, num_folds)

    # Store average accuracy and f1 for each k
    accuracy_scores = []
    f1_scores = []

    for k in k_values:
        print(f"\nRunning for k = {k}")
        acc_list, f1_list = [], []

        for i in range(num_folds):
            test_df = folds[i]
            train_df = pd.concat(folds[:i] + folds[i+1:], ignore_index=True)

            X_train, X_test, y_train, y_test = preprocess_data(
                train_df, test_df, categorical_cols, numeric_cols, target_column=target_column)

            y_pred = knn(X_train, y_train, X_test, k)

            y_pred_flat = y_pred.flatten()
            y_test_flat = y_test.flatten()

            metrics = calculate_metrics(y_test_flat, y_pred_flat)
            acc_list.append(metrics['Accuracy'] * 100)
            f1_list.append(metrics['F1'] * 100)

        avg_acc = np.mean(acc_list)
        avg_f1 = np.mean(f1_list)

        accuracy_scores.append(avg_acc)
        f1_scores.append(avg_f1)

        print(f"Average Accuracy at k={k}: {avg_acc:.2f}%")
        print(f"Average F1 Score at k={k}: {avg_f1:.2f}%")

    createGraph(k_values, accuracy_scores, 'Accuracy vs. k', 'Average Accuracy (%)', 'Accuracy')
    createGraph(k_values, f1_scores, 'F1 Score vs. k', 'Average F1 Score (%)', 'F1 Score')

# X, y = load_dataset()
# y = y.reshape(-1, 1)  # Ensure y is 2D
# df = pd.DataFrame(X)
# categorical_cols = []  # All features are numeric
# numeric_cols = list(df.columns[:-1])
# run(X, y, categorical_cols, numeric_cols, target_column="label")

# X, y = load_from_csv("../data/parkinsons.csv", target_column="Diagnosis")
# df = pd.DataFrame(X)
# categorical_cols = []
# numeric_cols = [col for col in df.columns if col != "Diagnosis"]
# run(X, y, categorical_cols, numeric_cols, target_column="Diagnosis")

# X, y = load_from_csv("../data/rice.csv", target_column="label")
# df = pd.DataFrame(X)
# categorical_cols = []
# numeric_cols = [col for col in df.columns if col != "label"]
# run(X, y, categorical_cols, numeric_cols, target_column="label")

# X_array, y = load_from_csv("../data/credit_approval.csv", target_column="label")
# df = pd.read_csv("../data/credit_approval.csv")
# feature_cols = df.drop(columns=["label"]).columns.tolist()
# X = pd.DataFrame(X_array, columns=feature_cols)
# categorical_cols = ["attr1_cat", "attr4_cat", "attr5_cat", "attr6_cat", 
#                     "attr7_cat", "attr9_cat", "attr10_cat", "attr11_cat", 
#                     "attr12_cat", "attr13_cat"]
# numeric_cols = ["attr2_num", "attr3_num", "attr8_num", "attr14_num", "attr15_num"]
# run(X, y, categorical_cols, numeric_cols, target_column="label")

X_array, y = load_student_data_from_csv()
df = pd.read_csv("../data/student_performance_dataset.csv")
feature_cols = df.drop(columns=["GradeClass", "StudentID"]).columns.tolist()
X = pd.DataFrame(X_array, columns=feature_cols)
categorical_cols = [
    'Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring',
    'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'
]

numeric_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']
run(X, y, categorical_cols, numeric_cols, target_column="GradeClass")