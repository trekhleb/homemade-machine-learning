"""Add sinusoid features to the features set"""

import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """Extends data set with sinusoid features.

    Returns a new feature array with more features, comprising of
    sin(x).

    :param dataset: data set.
    :param sinusoid_degree: multiplier for sinusoid parameter multiplications
    """

    # Create sinusoids matrix.
    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))

    # Generate sinusoid features of specified degree.
    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    # Return generated sinusoidal features.
    return sinusoids
