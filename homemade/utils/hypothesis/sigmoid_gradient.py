from .sigmoid import sigmoid


def sigmoid_gradient(z):
    """Computes the gradient of the sigmoid function evaluated at z."""

    return sigmoid(z) * (1 - sigmoid(z))
