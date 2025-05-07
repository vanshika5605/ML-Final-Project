import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict
from decision_tree import DecisionTreeClassifier, TreeNode
import matplotlib.pyplot as plt

# random.seed(42)
# np.random.seed(42)

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

def calculate_metrics(y_true, y_pred):
    """
    Calculates accuracy, precision, recall, and F1-score based on true and predicted labels.
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_metrics = {
        label: {'TP': 0, 'FP': 0, 'FN': 0}
        for label in labels
    }

    # Populate TP, FP, FN counts for each class
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

def train_random_forest_on_folds(folds, ntree_values, feature_types, max_depth, target_column):
    """
    Trains a Random Forest using different values of ntree (number of trees),
    performing stratified k-fold cross-validation for each value and returning average metrics.
    """ 
    results = []

    k = len(folds)

    for ntree in ntree_values:
        accs, precs, recs, f1s = [], [], [], []
        print("ntree value ", ntree)

        # Perform k-fold cross-validation
        for i in range(k):
            test_fold = folds[i] # Current test fold
            train_folds = [folds[j] for j in range(k) if j != i] # Remaining folds as training data
            train_data = pd.concat(train_folds, ignore_index=True)
            
            # Separate features and labels
            X_test = test_fold.drop(columns=target_column).values
            y_test = test_fold[target_column].values

            preds_from_all_trees = []

            for tree_idx in range(ntree):
                # Bootstrap sampling from training data (sampling with replacement)
                # bootstrap_sample = train_data.sample(n=len(train_data), replace=True, random_state=i * ntree + tree_idx)
                bootstrap_sample = train_data.sample(n=len(train_data), replace=True, random_state=None)
                X_train = bootstrap_sample.drop(columns=target_column).values
                y_train = bootstrap_sample[target_column].values

                # Random feature selection: m = sqrt(total features)
                m = int(np.sqrt(len(feature_types)))
                tree = DecisionTreeClassifier(max_depth, feature_types, m)

                # Train a tree and get predictions
                tree.fit(X_train, y_train)
                preds = tree.predict(X_test)
                preds_from_all_trees.append(preds)

            # Perform majority voting across all trees
            all_preds = np.array(preds_from_all_trees) # Shape: (ntree, num_samples)
            y_pred = []
            for j in range(X_test.shape[0]):
                votes = all_preds[:, j] # Predictions for sample j from all trees
                y_pred.append(Counter(votes).most_common(1)[0][0])

            # Metrics
            metrics = calculate_metrics(y_test, np.array(y_pred))
            accs.append(metrics['Accuracy'])
            precs.append(metrics['Precision'])
            recs.append(metrics['Recall'])
            f1s.append(metrics['F1'])

        results.append({
            'ntree': ntree,
            'Accuracy': np.mean(accs),
            'Precision': np.mean(precs),
            'Recall': np.mean(recs),
            'F1': np.mean(f1s)
        })

    return results # List of dictionaries with averaged metrics for each ntree