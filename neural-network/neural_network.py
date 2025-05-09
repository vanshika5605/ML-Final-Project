import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from utils import sigmoid, sigmoid_derivative, one_hot_encode
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], alpha=0.1, lam=0.0):
        self.layer_sizes = layer_sizes
        self.alpha = alpha
        self.lam = lam
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i] + 1) * 0.1
            weights.append(w)
        return weights

    def forward(self, X):
        a, z = [X], []
        for W in self.weights:
            X = np.insert(a[-1], 0, 1, axis=1)
            z_curr = np.dot(X, W.T)
            a_curr = sigmoid(z_curr)
            z.append(z_curr)
            a.append(a_curr)
        return a, z

    def compute_cost(self, y_true, y_pred, regularized=True):
        m = y_true.shape[0]
        epsilon = 1e-10
        cost = -np.sum(y_true * np.log(y_pred + epsilon)) / m

        if regularized:
            reg = sum(np.sum(w[:, 1:] ** 2) for w in self.weights)
            cost += (self.lam / (2 * m)) * reg

        return cost

    def backward(self, a, z, y_true):
        m = y_true.shape[0]
        deltas = [a[-1] - y_true]

        for i in reversed(range(len(self.weights) - 1)):
            a_curr = np.insert(a[i+1], 0, 1, axis=1)
            W = self.weights[i+1]
            delta = deltas[0].dot(W[:, 1:]) * sigmoid_derivative(a[i+1])
            deltas.insert(0, delta)

        grads = []
        for i in range(len(self.weights)):
            a_input = np.insert(a[i], 0, 1, axis=1)
            grad = deltas[i].T.dot(a_input) / m
            grad[:, 1:] += (self.lam / m) * self.weights[i][:, 1:]
            grads.append(grad)
        return grads

    def update_weights(self, grads):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * grads[i]

    def fit(self, X, y, epochs=1000, verbose=True):
        history = []
        for epoch in range(epochs):
            a, z = self.forward(X)
            cost = self.compute_cost(y, a[-1])
            history.append(cost)

            grads = self.backward(a, z, y)
            self.update_weights(grads)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
        return history

    def predict(self, X):
        a, _ = self.forward(X)
        return np.argmax(a[-1], axis=1)

def evaluate_model(X, y, architecture, lam=0.0, alpha=0.01, verbose=False):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs, f1s = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        y_train_encoded = one_hot_encode(y_train)
        nn = NeuralNetwork(architecture, alpha=alpha, lam=lam)
        nn.fit(X_train, y_train_encoded, epochs=1000, verbose=False)

        preds = nn.predict(X_test)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average='macro'))

    return np.mean(accs), np.mean(f1s)

def epoch_based_learning_curve(X, y, architecture, lam, alpha=0.1, epochs=30):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # One-hot encode labels
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)

    nn = NeuralNetwork(architecture, alpha=alpha, lam=lam)

    test_costs = []
    x_axis = []

    m = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        # Train one epoch
        nn.fit(X_train, y_train_encoded, epochs=100, verbose=False)

        # Evaluate cost on test set
        a_test, _ = nn.forward(X_test)
        cost = nn.compute_cost(y_test_encoded, a_test[-1])

        test_costs.append(cost)
        x_axis.append(epoch * m)

        print(f"Epoch {epoch} → Seen {epoch * m} samples → Test Cost J = {cost:.4f}")

    # Plotting the test cost curve
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, test_costs, marker='o')
    plt.title(f"Epoch-based Learning Curve (α={alpha}, λ={lam})")
    plt.xlabel("Training Examples Seen")
    plt.ylabel("Test Cost J")
    plt.grid(True)
    plt.tight_layout()
    plt.show()