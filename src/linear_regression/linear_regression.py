"""Linear Regression Module"""

import numpy as np


class LinearRegression:
    """Linear Regression Class"""

    def __init__(self, training_set, labels):
        """Linear regression constructor.

        :param training_set: training set.
        :param labels: training set outputs (correct values).
        """

        # Calculate the number of training examples and features.
        num_examples = training_set.shape[0]
        num_features = training_set.shape[1]

        # Normalize features.
        (
            training_set_normalized,
            features_mean,
            features_deviation
        ) = LinearRegression.normalize_features(training_set)

        # Add a column of ones to X.
        training_set_normalized = np.hstack((np.ones((num_examples, 1)), training_set_normalized))

        self.training_set = training_set_normalized
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation

        # Initialize model parameters.
        self.theta = np.zeros((num_features + 1, 1))

    def train(self, alpha, lambda_param, num_iterations):
        """Trains linear regression.

        :param alpha: learning rate (the size of the step for gradient descent)
        :param lambda_param: regularization parameter
        :param num_iterations: number of gradient descent iterations.
        """

        # Run gradient descent.
        cost_history = self.gradient_descent(alpha, lambda_param, num_iterations)

        return self.theta, cost_history

    def gradient_descent(self, alpha, lambda_param, num_iterations):
        """Gradient descent.

        It calculates what steps (deltas) should be taken for each theta parameter in
        order to minimize the cost function.

        :param alpha: learning rate (the size of the step for gradient descent)
        :param lambda_param: regularization parameter
        :param num_iterations: number of gradient descent iterations.
        """

        # Initialize J_history with zeros.
        cost_history = []

        for iteration in range(num_iterations):
            # Perform a single gradient step on the parameter vector theta.
            self.gradient_step(alpha, lambda_param)

            # Save the cost J in every iteration.
            cost_history.append(self.cost_function(lambda_param))

        return cost_history

    def gradient_step(self, alpha, lambda_param):
        """Gradient step.

        Function performs one step of gradient descent for theta parameters.

        :param alpha: learning rate (the size of the step for gradient descent)
        :param lambda_param: regularization parameter
        """

        # Calculate the number of training examples and features.
        num_examples = self.training_set.shape[0]

        # Predictions of hypothesis on all m examples.
        predictions = self.hypothesis()

        # The difference between predictions and actual values for all m examples.
        delta = predictions - self.labels

        # Calculate regularization parameter.
        reg_param = 1 - alpha * lambda_param / num_examples

        # Create theta shortcut.
        theta = self.theta

        # Vectorized version of gradient descent.
        theta = theta * reg_param - alpha * (1 / num_examples) * (delta.T @ self.training_set).T
        # We should NOT regularize the parameter theta_zero.
        theta[1] = theta[1] - alpha * (1 / num_examples) * (self.training_set[:, 0].T @ delta).T

        self.theta = theta

    def cost_function(self, lambda_param):
        """Cost function.

        It shows how accurate our model is based on current model parameters.

        :param lambda_param: regularization parameter
        """

        # Calculate the number of training examples and features.
        num_examples = self.training_set.shape[0]

        # Get the difference between predictions and correct output values.
        delta = self.hypothesis() - self.labels

        # Calculate regularization parameter.
        # Remember that we should not regularize the parameter theta_zero.
        theta_cut = self.theta[1:, 0]
        reg_param = lambda_param * (theta_cut.T @ theta_cut)

        # Calculate current predictions cost.
        cost = (1 / 2 * num_examples) * (delta.T @ delta + reg_param)

        # Let's extract cost value from the one and only cost numpy matrix cell.
        return cost[0][0]

    def hypothesis(self):
        """Hypothesis function.

        It predicts the output values y based on the input values X and model parameters.

        :return: predictions made by model based on provided theta.
        """

        predictions = self.training_set @ self.theta

        return predictions

    def predict(self, data_set):
        """Predict the output for data_set input based on trained theta values"""
        (data_set_normalized, mean, deviation) = self.normalize_features(data_set)
        predictions = data_set_normalized @ self.theta
        return predictions

    @staticmethod
    def normalize_features(training_set):
        """Normalize input features.

        Normalizes the features in x. Returns a normalized version of x where the mean value of
        each feature is 0 and the standard deviation is 1.

        :param training_set: training set of features.
        :return: normalized set of features.
        """

        # Copy original array to prevent it from changes.
        training_set_normalized = np.copy(training_set)

        # Get average values for each feature (column) in X.
        features_mean = np.mean(training_set, 0)

        # Calculate the standard deviation for each feature.
        features_deviation = np.std(training_set, 0)

        # Subtract mean values from each feature (column) of every example (row)
        # to make all features be spread around zero.
        training_set_normalized -= features_mean

        # Normalize each feature values for each example so that all features
        # are close to [-1:1] boundaries.
        training_set_normalized /= features_deviation

        return training_set_normalized, features_mean, features_deviation
