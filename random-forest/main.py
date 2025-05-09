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
from load_dataset import load_dataset, load_from_csv, load_student_data_from_csv 

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


def run_classifier(X, y, target_column, feature_types, max_depth):
    # Define ntree values and fixed max depth
    ntree_values = [1, 5, 10, 20, 30, 40, 50]

    # Get folds
    folds = stratified_k_fold(X, y, target_column, k=10)

    # Train and evaluate
    results = train_random_forest_on_folds(folds, ntree_values, feature_types, max_depth, target_column)

    # Extract metric lists
    ntree_list = [res['ntree'] for res in results]
    accuracy_list = [res['Accuracy'] for res in results]
    f1_list = [res['F1'] for res in results]

    # Print Accuracy and F1 for each ntree
    print("ntree\tAccuracy\tF1-score")
    for res in results:
        print(f"{res['ntree']}\t{res['Accuracy']:.4f}\t\t{res['F1']:.4f}")

    # Collect metrics for plotting
    metrics = {
        'Accuracy': accuracy_list,
        'Precision': [res['Precision'] for res in results],
        'Recall': [res['Recall'] for res in results],
        'F1-score': f1_list
    }

    # Plot all metrics in a single figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()  # Flatten 2D array of axes

    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        axes[i].plot(ntree_list, metric_values, marker='o')
        axes[i].set_title(f'{metric_name} vs Number of Trees (ntree)')
        axes[i].set_xlabel('ntree')
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

# To run the classifier for different datasets

# X, y = load_dataset()
# run_classifier(X, y, "label", ['numerical'] * 64, 13)

# X, y = load_from_csv("../data/parkinsons.csv", "Diagnosis")
# run_classifier(X, y, "Diagnosis", ['numerical'] * 22, 13)

# X, y = load_from_csv("../data/rice.csv", "label")
# run_classifier(X, y, "label", ['numerical'] * 7, 7)

# feature_types = ['categorical','numerical','numerical','categorical','categorical','categorical','categorical','numerical','categorical','categorical','categorical','categorical','categorical','numerical','numerical']
# X, y = load_from_csv("../data/credit_approval.csv", "label")
# run_classifier(X, y, "label", feature_types, 15)

feature_types = [
    'numerical',    # Age
    'categorical',  # Gender
    'categorical',  # Ethnicity
    'categorical',  # ParentalEducation
    'numerical',    # StudyTimeWeekly
    'numerical',    # Absences
    'categorical',  # Tutoring
    'categorical',  # ParentalSupport
    'categorical',  # Extracurricular
    'categorical',  # Sports
    'categorical',  # Music
    'categorical',  # Volunteering
    'numerical',    # GPA
]
X, y = load_student_data_from_csv()
run_classifier(X, y, "GradeClass", feature_types, 7)


# To test the classifier for different maximal_depths

# X, y = load_dataset()
# test_depths(X, y, ['numerical'] * 64, "label")

# X, y = load_from_csv("../data/parkinsons.csv", "Diagnosis")
# test_depths(X, y, ['numerical'] * 22, "Diagnosis")

# X, y = load_from_csv("../data/rice.csv", "label")
# test_depths(X, y, ['numerical'] * 7, "label")

# feature_types = ['categorical','numerical','numerical','categorical','categorical','categorical','categorical','numerical','categorical','categorical','categorical','categorical','categorical','numerical','numerical']
# X, y = load_from_csv("../data/credit_approval.csv", "label")
# test_depths(X, y, feature_types, "label")

# feature_types = [
#     'numerical',    # Age
#     'categorical',  # Gender
#     'categorical',  # Ethnicity
#     'categorical',  # ParentalEducation
#     'numerical',    # StudyTimeWeekly
#     'numerical',    # Absences
#     'categorical',  # Tutoring
#     'categorical',  # ParentalSupport
#     'categorical',  # Extracurricular
#     'categorical',  # Sports
#     'categorical',  # Music
#     'categorical',  # Volunteering
#     'numerical',    # GPA
# ]
# X, y = load_student_data_from_csv()

# test_depths(X, y, feature_types, "GradeClass")
