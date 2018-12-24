"""Add sinusoid features to the features set"""

import numpy as np


def add_sinusoids(dataset, sinusoid_degree):
    """Extends data set with sinusoid features.

    Returns a new feature array with more features, comprising of
    sin(x).

    :param dataset: data set.
    :param sinusoid_degree: multiplier for sinusoid parameter multiplications
    """

    sinusoids = np.empty((dataset.shape[0], 0))

    for degree in range(1, sinusoid_degree):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    return sinusoids
