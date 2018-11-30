import numpy as np


def add_sinusoids(x, sinusoid_degree):
    """Extends data set with sinusoid features.

    Returns a new feature array with more features, comprising of
    sin(x).

    :param x: data set.
    :param sinusoid_degree: multiplier for sinusoid parameter multiplications
    """

    sinusoids = np.empty((x.shape[0], 0))

    for degree in range(1, sinusoid_degree):
        sinusoid_features = np.sin(degree * x)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    return sinusoids
