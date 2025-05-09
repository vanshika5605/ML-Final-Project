import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def one_hot_encode(y):
    n_classes = np.max(y) + 1
    return np.eye(n_classes)[y]

