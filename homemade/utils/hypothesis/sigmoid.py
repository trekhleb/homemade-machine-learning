"""Sigmoid function"""

import numpy as np


def sigmoid(matrix):
    """Applies sigmoid function to NumPy matrix"""

    return 1 / (1 + np.exp(-matrix))
