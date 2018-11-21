"""Linear Regression Module"""

import numpy as np


class LinearRegression:
    """Linear Regression Class"""

    def __init__(self, data, labels):
        """Linear regression constructor.

        :param data: training set.
        :param labels: training set outputs (correct values).
        """

        # Normalize features and add ones column.
        (
            data_processed,
            features_mean,
            features_deviation
        ) = LinearRegression.prepare_data(data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation

        # Initialize model parameters.
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

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
            cost_history.append(self.cost_function(self.data, self.labels, lambda_param))

        return cost_history

    def gradient_step(self, alpha, lambda_param):
        """Gradient step.

        Function performs one step of gradient descent for theta parameters.

        :param alpha: learning rate (the size of the step for gradient descent)
        :param lambda_param: regularization parameter
        """

        # Calculate the number of training examples.
        num_examples = self.data.shape[0]

        # Predictions of hypothesis on all m examples.
        predictions = LinearRegression.hypothesis(self.data, self.theta)

        # The difference between predictions and actual values for all m examples.
        delta = predictions - self.labels

        # Calculate regularization parameter.
        reg_param = 1 - alpha * lambda_param / num_examples

        # Create theta shortcut.
        theta = self.theta

        # Vectorized version of gradient descent.
        theta = theta * reg_param - alpha * (1 / num_examples) * (delta.T @ self.data).T
        # We should NOT regularize the parameter theta_zero.
        theta[1] = theta[1] - alpha * (1 / num_examples) * (self.data[:, 0].T @ delta).T

        self.theta = theta

    def get_cost(self, data, labels, lambda_param):
        """Get the cost value for specific data set.

        :param data: the set of training or test data.
        :param labels: training set outputs (correct values).
        :param lambda_param: regularization parameter
        """

        data_processed = LinearRegression.prepare_data(data)[0]

        return self.cost_function(data_processed, labels, lambda_param)

    def cost_function(self, data, labels, lambda_param):
        """Cost function.

        It shows how accurate our model is based on current model parameters.

        :param data: the set of training or test data.
        :param labels: training set outputs (correct values).
        :param lambda_param: regularization parameter
        """

        # Calculate the number of training examples and features.
        num_examples = data.shape[0]

        # Get the difference between predictions and correct output values.
        delta = LinearRegression.hypothesis(data, self.theta) - labels

        # Calculate regularization parameter.
        # Remember that we should not regularize the parameter theta_zero.
        theta_cut = self.theta[1:, 0]
        reg_param = lambda_param * (theta_cut.T @ theta_cut)

        # Calculate current predictions cost.
        cost = (1 / 2 * num_examples) * (delta.T @ delta + reg_param)

        # Let's extract cost value from the one and only cost numpy matrix cell.
        return cost[0][0]

    def predict(self, data):
        """Predict the output for data_set input based on trained theta values

        :param data: training set of features.
        """

        # Normalize features and add ones column.
        data_processed = LinearRegression.prepare_data(data)[0]

        # Do predictions using model hypothesis.
        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions

    @staticmethod
    def hypothesis(data, theta):
        """Hypothesis function.

        It predicts the output values y based on the input values X and model parameters.

        :param data: data set for what the predictions will be calculated.
        :param theta: model params.
        :return: predictions made by model based on provided theta.
        """

        predictions = data @ theta

        return predictions

    @staticmethod
    def prepare_data(data):
        """Prepares data set for training on prediction"""

        # Calculate the number of examples.
        num_examples = data.shape[0]

        # Normalize data set.
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = LinearRegression.normalize_features(data)

        # Add a column of ones to X.
        data_processed = np.hstack((np.ones((num_examples, 1)), data_normalized))

        return data_processed, features_mean, features_deviation

    @staticmethod
    def normalize_features(data):
        """Normalize input features.

        Normalizes the features in x. Returns a normalized version of x where the mean value of
        each feature is 0 and deviation is 1.

        :param data: training set of features.
        :return: normalized set of features.
        """

        # Copy original array to prevent it from changes.
        data_normalized = np.copy(data)

        # Get average values for each feature (column) in X.
        features_mean = np.mean(data, 0)

        # Calculate the standard deviation for each feature.
        features_deviation = np.std(data, 0)

        # Subtract mean values from each feature (column) of every example (row)
        # to make all features be spread around zero.
        data_normalized -= features_mean

        # Normalize each feature values for each example so that all features
        # are close to [-1:1] boundaries.
        data_normalized /= features_deviation

        return data_normalized, features_mean, features_deviation
