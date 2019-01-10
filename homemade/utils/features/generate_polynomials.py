"""Add polynomial features to the features set"""

import numpy as np
from .normalize import normalize


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """Extends data set with polynomial features of certain degree.

    Returns a new feature array with more features, comprising of
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.

    :param dataset: dataset that we want to generate polynomials for.
    :param polynomial_degree: the max power of new features.
    :param normalize_data: flag that indicates whether polynomials need to normalized or not.
    """

    # Split features on two halves.
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    # Extract sets parameters.
    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    # Check if two sets have equal amount of rows.
    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    # Check if at list one set has features.
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    # Replace empty set with non-empty one.
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # Make sure that sets have the same number of features in order to be able to multiply them.
    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    # Create polynomials matrix.
    polynomials = np.empty((num_examples_1, 0))

    # Generate polynomial features of specified degree.
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    # Normalize polynomials if needed.
    if normalize_data:
        polynomials = normalize(polynomials)[0]

    # Return generated polynomial features.
    return polynomials
