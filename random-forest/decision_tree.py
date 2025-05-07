import numpy as np
from collections import defaultdict
import random
from sklearn.utils import shuffle

# Defining a node structure for the tree
class TreeNode:
    def __init__(self, feature=None, threshold=None, categories=None, children=None, value=None):
        self.feature = feature        # Feature index we split on
        self.threshold = threshold    # Only used for numerical splits: Threshold value for numerical splits
        self.categories = categories  # Dictionary to store a mapping of category values to child node indices
        self.children = children      # List of child nodes
        self.value = value            # Value used for leaf nodes (predicted class)

    # If the node has a value it is a leaf node
    def is_leaf(self):
        return self.value is not None

# Defining a class for the decision tree
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, feature_types=None, m_features=None):
        self.max_depth = max_depth  # Max depth of the tree (for pruning)
        self.feature_types = feature_types # List describing each feature as 'categorical' or 'numerical'
        self.m_features = m_features  # Number of features to consider at each split

    # To calculate the entropy for a partition
    def calculate_entropy(self, y):
        entropy = 0

        # get the unique target values
        labels = np.unique(y)

        for label in labels:
            # get all the elements with the current value in y
            elements = y[y==label]

            # calculate the probability of the current value
            p = len(elements)/len(y)

            # calculate the entropy
            entropy += -p*np.log2(p)

        return entropy

    # To calculate information gain for a split
    def calculate_information_gain(self, target, category_indices):
        # Calculating entropy of the parent
        parent_entropy = self.calculate_entropy(target)

        # Calculate weighted entropy for each category
        weighted_entropy = 0
        for category, indices in category_indices.items():
            # Making sure we have atleast some samples for each category
            if len(indices) > 0:
                category_weight = len(indices) / len(target)
                category_entropy = self.calculate_entropy(target[indices])
                weighted_entropy += category_weight * category_entropy

        # Calculating information gain
        info_gain = parent_entropy - weighted_entropy
        return info_gain

    # Split data based on a categorical feature
    def split_data(self, dataset, feature):
        # Get unique categories in the feature data
        categories = np.unique(dataset[:, feature])

        # Initialize a dictionary to store indices of data points for each category
        category_indices = defaultdict(list)

        # Assign each data point to its category, thus creating a list of indices for each category
        for idx, row in enumerate(dataset):
            category = row[feature]
            category_indices[category].append(idx)

        # Convert the lists of indices to numpy arrays
        for category in category_indices:
            category_indices[category] = np.array(category_indices[category])

        return category_indices, categories

    # To find the feature that splits the data with the largest information gain
    def get_best_split(self, dataset, target, feature_indices):
        best_gain = -1
        best_feature = None
        best_split = None
        best_categories = None
        best_category_indices = None

        # Apply m-feature random selection for Random Forest
        if self.m_features is not None and self.m_features < len(feature_indices):
            feature_indices = random.sample(feature_indices, self.m_features)

        random.shuffle(feature_indices)

        for feat_idx in feature_indices:
            feature_type = self.feature_types[feat_idx]

            if feature_type == 'categorical':
                # Handle categorical attribute
                category_indices, categories = self.split_data(dataset, feat_idx)

                if len(category_indices) < 2:
                    continue

                gain = self.calculate_information_gain(target, category_indices)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_split = None
                    best_category_indices = category_indices
                    best_categories = categories

            elif feature_type == 'numerical':
                # Sort feature values and target
                feature_values = dataset[:, feat_idx].astype(float)
                sorted_indices = np.argsort(feature_values)
                sorted_vals = feature_values[sorted_indices]
                sorted_target = target[sorted_indices]

                # Get thresholds where the class label changes
                thresholds = []
                for i in range(1, len(sorted_vals)):
                    if sorted_target[i] != sorted_target[i - 1]:
                        threshold = (sorted_vals[i] + sorted_vals[i - 1]) / 2.0
                        thresholds.append(threshold)

                # Evaluate each threshold for info gain
                for threshold in thresholds:
                    left_indices = np.where(feature_values <= threshold)[0]
                    right_indices = np.where(feature_values > threshold)[0]

                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue

                    split_indices = {'<=': left_indices, '>': right_indices}
                    gain = self.calculate_information_gain(target, split_indices)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feat_idx
                        best_split = threshold
                        best_category_indices = split_indices
                        best_categories = ['<=', '>']

        return best_feature, best_category_indices, best_categories, best_gain, best_split

    # To get the most common value in a leaf node
    def leaf_value(self, y):
        y = list(y)
        if not y:
            return None  # Handle empty list case
        most_common = max(set(y), key=y.count)
        return most_common

    # To create the decision tree
    def create_tree(self, X, y, features, depth=0):
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        # Base cases: all same label, no features left, or max depth reached
        if n_labels == 1 or not features or (self.max_depth is not None and depth >= self.max_depth):
            return TreeNode(value=self.leaf_value(y))

        # Get best split
        best_feature, category_indices, categories, best_gain, best_threshold = self.get_best_split(X, y, features)

        if best_gain > 0 and best_feature is not None:
            children = []
            feature_type = self.feature_types[best_feature]
            remaining_features = [f for f in features if f != best_feature]

            if feature_type == 'categorical':
                categories_dict = {}
                for i, category in enumerate(categories):
                    indices = category_indices.get(category, np.array([]))
                    if len(indices) > 0:
                        child = self.create_tree(X[indices], y[indices], remaining_features, depth + 1)
                    else:
                        child = TreeNode(value=self.leaf_value(y))
                    children.append(child)
                    categories_dict[category] = i

                return TreeNode(feature=best_feature, categories=categories_dict, children=children)

            elif feature_type == 'numerical':
                left_indices = category_indices['<=']
                right_indices = category_indices['>']

                left_child = self.create_tree(X[left_indices], y[left_indices], remaining_features, depth + 1)
                right_child = self.create_tree(X[right_indices], y[right_indices], remaining_features, depth + 1)

                return TreeNode(feature=best_feature, threshold=best_threshold, children=[left_child, right_child])

        # If no good split, return a leaf
        return TreeNode(value=self.leaf_value(y))

    # To fit the data to the decision tree
    def fit(self, X, y):
        n_features = X.shape[1]
        if self.feature_types is None:
            raise ValueError("feature_types must be provided")
        if len(self.feature_types) != n_features:
            raise ValueError("feature_types must match number of features")
        self.root = self.create_tree(X, y, features=list(range(n_features)))
        return self

    # To predict the value for a sample x starting from a node in the tree
    def predict_value(self, x, node):
        if node.is_leaf():
            return node.value

        feat = node.feature
        feature_type = self.feature_types[feat]

        if node.threshold is not None and feature_type == 'numerical':
            try:
                value = float(x[feat])
            except:
                # If value can't be converted, default to left child or node value
                return node.children[0].value if node.children else node.value
            if value <= node.threshold:
                return self.predict_value(x, node.children[0])
            else:
                return self.predict_value(x, node.children[1])

        else:
            category = x[feat]
            if category not in node.categories:
                return self.predict_value(x, node.children[0]) if node.children else node.value
            child_idx = node.categories[category]
            if child_idx < len(node.children):
                return self.predict_value(x, node.children[child_idx])
            else:
                return node.value

    # Predict values for each sample on X
    def predict(self, X):
        if not hasattr(self, 'root') or self.root is None:
            raise ValueError("Model not fitted yet. Call 'fit' before using 'predict'.")

        pred_values = [self.predict_value(x, self.root) for x in X]
        return np.array(pred_values)