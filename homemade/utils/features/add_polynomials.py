"""Add polynomial features to the features set"""

import numpy as np


def add_polynomials(dataset_1, dataset_2, polynomial_degree):
    """Extends data set with polynomial features of certain degree.

    Returns a new feature array with more features, comprising of
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.

    :param dataset_1: first data set.
    :param dataset_2: second data set.
    :param polynomial_degree: the max power of new features.
    """

    polynomials = np.empty((dataset_1.shape[0], 0))

    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    return polynomials
