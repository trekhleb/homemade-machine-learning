import numpy as np


def sigmoid(z):
    """Applies sigmoid function to NumPy matrix"""
    return 1 / (1 + np.exp(-1 * z))
