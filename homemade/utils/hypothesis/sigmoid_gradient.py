"""Sigmoid gradient function"""

from .sigmoid import sigmoid


def sigmoid_gradient(matrix):
    """Computes the gradient of the sigmoid function evaluated at z."""

    return sigmoid(matrix) * (1 - sigmoid(matrix))
