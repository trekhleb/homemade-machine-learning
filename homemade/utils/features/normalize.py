import numpy as np


def normalize(features):
    """Normalize features.

    Normalizes input features X. Returns a normalized version of X where the mean value of
    each feature is 0 and deviation is close to 1.

    :param features: set of features.
    :return: normalized set of features.
    """

    # Copy original array to prevent it from changes.
    features_normalized = np.copy(features)

    # Get average values for each feature (column) in X.
    features_mean = np.mean(features, 0)

    # Calculate the standard deviation for each feature.
    features_deviation = np.std(features, 0)

    # Subtract mean values from each feature (column) of every example (row)
    # to make all features be spread around zero.
    features_normalized -= features_mean

    # Normalize each feature values for each example so that all features
    # are close to [-1:1] boundaries.
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
