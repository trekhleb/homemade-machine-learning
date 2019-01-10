"""Linear Regression Module"""

# Import dependencies.
import numpy as np
from ..utils.features import prepare_for_training


class LinearRegression:
    # pylint: disable=too-many-instance-attributes
    """Linear Regression Class"""

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        # pylint: disable=too-many-arguments
        """Linear regression constructor.

        :param data: training set.
        :param labels: training set outputs (correct values).
        :param polynomial_degree: degree of additional polynomial features.
        :param sinusoid_degree: multipliers for sinusoidal features.
        :param normalize_data: flag that indicates that features should be normalized.
        """

        # Normalize features and add ones column.
        (
            data_processed,
            features_mean,
            features_deviation
        ) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # Initialize model parameters.
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, lambda_param=0, num_iterations=500):
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

        for _ in range(num_iterations):
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
        theta[0] = theta[0] - alpha * (1 / num_examples) * (self.data[:, 0].T @ delta).T

        self.theta = theta

    def get_cost(self, data, labels, lambda_param):
        """Get the cost value for specific data set.

        :param data: the set of training or test data.
        :param labels: training set outputs (correct values).
        :param lambda_param: regularization parameter
        """

        data_processed = prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )[0]

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
        data_processed = prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )[0]

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
