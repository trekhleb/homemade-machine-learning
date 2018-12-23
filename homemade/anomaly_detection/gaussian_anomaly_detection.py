"""Anomaly Detection Module"""

import numpy as np
import math


class GaussianAnomalyDetection:
    """GaussianAnomalyDetection Class"""

    def __init__(self, data):
        """GaussianAnomalyDetection constructor"""

        # Estimate Gaussian distribution.
        (self.mu, self.sigma_squared) = GaussianAnomalyDetection.estimate_gaussian(data)

        # Save training data.
        self.data = data

    def multivariate_gaussian(self, data):
        """Computes the probability density function of the multivariate gaussian distribution"""

        mu = self.mu
        sigma_squared = self.sigma_squared

        # Get number of training sets and features.
        (num_examples, num_features) = data.shape

        # nit probabilities matrix.
        probabilities = np.ones((num_examples, 1))

        # Go through all training examples and through all features.
        for example_index in range(num_examples):
            for feature_index in range(num_features):
                # Calculate the power of e.
                e_power_dividend = (data[example_index, feature_index] - mu[feature_index]) ** 2
                e_power_divider = 2 * sigma_squared[feature_index]
                e_power = -1 * e_power_dividend / e_power_divider

                # Calculate the prefix multiplier.
                probability_prefix = 1 / math.sqrt(2 * math.pi * sigma_squared[feature_index])

                # Calculate the probability for the current feature of current example.
                probability = probability_prefix * (math.e ** e_power)
                probabilities[example_index] *= probability

        # Return probabilities for all training examples.
        return probabilities

    @staticmethod
    def estimate_gaussian(data):
        """This function estimates the parameters of a Gaussian distribution using the data in X."""

        # Get number of features and number of examples.
        (num_examples, num_features) = data.shape

        # Estimate Gaussian parameters mu and sigma_squared for every feature.
        mu = (1 / num_examples) * np.sum(data, axis=0)
        sigma_squared = (1 / num_examples) * np.sum((data - mu) ** 2, axis=0)

        # Return Gaussian parameters.
        return mu, sigma_squared

    @staticmethod
    def select_threshold(labels, probabilities):
        """Finds the best threshold (epsilon) to use for selecting outliers"""

        best_epsilon = 0
        best_f1 = 0

        # History data to build the plots.
        precision_history = []
        recall_history = []
        f1_history = []

        # Calculate the epsilon steps.
        min_probability = np.min(probabilities)
        max_probability = np.max(probabilities)
        step_size = (max_probability - min_probability) / 1000

        # Go through all possible epsilons and pick the one with the highest f1 score.
        for epsilon in np.arange(min_probability, max_probability, step_size):
            predictions = probabilities < epsilon

            # The number of false positives: the ground truth label says it’s not
            # an anomaly, but our algorithm incorrectly classified it as an anomaly.
            false_positives = np.sum((predictions == 1) & (labels == 0))

            # The number of false negatives: the ground truth label says it’s an anomaly,
            # but our algorithm incorrectly classified it as not being anomalous.
            false_negatives = np.sum((predictions == 0) & (labels == 1))

            # The number of true positives: the ground truth label says it’s an
            # anomaly and our algorithm correctly classified it as an anomaly.
            true_positives = np.sum((predictions == 1) & (labels == 1))

            # Prevent division by zero.
            if (true_positives + false_positives) == 0 or (true_positives + false_negatives) == 0:
                continue

            # Precision.
            precision = true_positives / (true_positives + false_positives)

            # Recall.
            recall = true_positives / (true_positives + false_negatives)

            # F1.
            f1 = 2 * precision * recall / (precision + recall)

            # Save history data.
            precision_history.append(precision)
            recall_history.append(recall)
            f1_history.append(f1)

            if f1 > best_f1:
                best_epsilon = epsilon
                best_f1 = f1

        return best_epsilon, best_f1, precision_history, recall_history, f1_history
