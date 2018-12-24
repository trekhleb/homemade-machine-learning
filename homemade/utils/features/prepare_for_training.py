"""Prepares the dataset for training"""

import math
import numpy as np
from .normalize import normalize
from .add_sinusoids import add_sinusoids
from .add_polynomials import add_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """Prepares data set for training on prediction"""

    # Calculate the number of examples.
    num_examples = data.shape[0]

    # Prevent original data from being modified.
    data_processed = np.copy(data)

    # Normalize data set.
    features_mean = 0
    features_deviation = 0
    if normalize_data:
        (
            data_processed,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

    # Add sinusoidal features to the dataset.
    if sinusoid_degree:
        data_processed = add_sinusoids(data_processed, sinusoid_degree)

    # Add polynomial features to data set.
    if polynomial_degree >= 2:
        current_features_num = data_processed.shape[1]
        middle_feature_index = math.floor(current_features_num / 2)

        # Split features on halves.
        features_split = np.split(data_processed, [middle_feature_index], axis=1)
        first_half = features_split[0]
        second_half = features_split[1]

        # Generate polynomials.
        data_processed = add_polynomials(first_half, second_half, polynomial_degree)

    # Add a column of ones to X.
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation
