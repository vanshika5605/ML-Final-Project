# How to run the code?
# Refer README.md

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier, TreeNode
from random_forest_classifier import stratified_k_fold, train_random_forest_on_folds
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from load_dataset import load_dataset, load_from_csv

# Function to test different maximal_depths for a dataset
def test_depths(X, y, feature_types, target_column):    
    # Define depths and ntree values
    depth_values = [1,3,5,7,9, 11, 13, 15, 17, 19, 21]
    ntree_values = [30, 40, 50]

    # Get folds
    folds = stratified_k_fold(X, y, target_column, k=10)

    # Store results
    metrics_per_depth = {metric: {ntree: [] for ntree in ntree_values} for metric in ['Accuracy', 'Precision', 'Recall', 'F1']}

    # Run training for each depth
    for depth in depth_values:
        print(f"\nFor max_depth {depth}")
        results = train_random_forest_on_folds(folds, ntree_values, feature_types, depth, target_column)

        for res in results:
            ntree = res['ntree']
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
                metrics_per_depth[metric][ntree].append(res[metric])

            print(f"ntree={ntree}: Acc={res['Accuracy']:.4f}, Prec={res['Precision']:.4f}, "
                f"Rec={res['Recall']:.4f}, F1={res['F1']:.4f}")

    # Plot all metrics in one figure (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Random Forest Metrics vs Max Depth (for various ntree)', fontsize=16)

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for metric, pos in zip(metric_names, positions):
        ax = axes[pos[0], pos[1]]
        for ntree in ntree_values:
            ax.plot(depth_values, metrics_per_depth[metric][ntree], label=f"ntree={ntree}", marker='o')
        ax.set_title(metric)
        ax.set_xlabel("Max Depth")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Function to test different ntree values for a dataset
def run_classifier(filename, feature_types, max_depth):
    # Load data
    df = pd.read_csv(filename)
    target_column = "label"

    X = df.drop(columns=target_column).values
    y = df[target_column].values

    # Define ntree values and fixed max depth
    ntree_values = [1, 5, 10, 20, 30, 40, 50]

    # Get folds
    folds = stratified_k_fold(X, y, target_column, k=10)

    # Train and evaluate
    results = train_random_forest_on_folds(folds, ntree_values, feature_types, max_depth, target_column)

    # Extract metric lists
    ntree_list = [res['ntree'] for res in results]
    metrics = {
        'Accuracy': [res['Accuracy'] for res in results],
        'Precision': [res['Precision'] for res in results],
        'Recall': [res['Recall'] for res in results],
        'F1-score': [res['F1'] for res in results]
    }

    # Plot each metric separately
    for metric_name, metric_values in metrics.items():
        plt.figure(figsize=(8, 5))
        plt.plot(ntree_list, metric_values, marker='o')
        plt.title(f'{metric_name} vs Number of Trees(ntree)')
        plt.xlabel('ntree')
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# To run the classifier for different datasets

# run_classifier("data/wdbc.csv", ['numerical'] * 30, 9)

# feature_types = ['categorical','categorical','categorical','categorical','categorical','numerical','numerical','numerical','numerical','categorical','categorical']
# run_classifier("data/loan.csv", feature_types, 5)

# run_classifier("data/raisin.csv", ['numerical'] * 7, 7)

# feature_types = ['categorical','categorical','numerical','numerical','numerical','numerical']
# run_classifier("data/titanic.csv", feature_types, 7)

# To test the classifier for different maximal_depths

# X, y = load_dataset()

# test_depths(X, y, ['numerical'] * 64, "label")

# X, y = load_from_csv("../data/parkinsons.csv", "Diagnosis")

# test_depths(X, y, ['numerical'] * 22, "Diagnosis")

# X, y = load_from_csv("../data/rice.csv", "label")

# test_depths(X, y, ['numerical'] * 7, "label")

feature_types = ['categorical','numerical','numerical','categorical','categorical','categorical','categorical','numerical','categorical','categorical','categorical','categorical','categorical','numerical','numerical']
X, y = load_from_csv("../data/credit_approval.csv", "label")
test_depths(X, y, feature_types, "label")
