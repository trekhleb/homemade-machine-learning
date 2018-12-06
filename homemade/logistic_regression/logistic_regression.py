import numpy as np
from scipy.optimize import minimize
from ..utils.features import prepare_for_training
from ..utils.hypothesis import sigmoid


class LogisticRegression:
    """Logistic Regression Class"""

    def __init__(self, data, labels):
        # Normalize features and add ones column.
        (
            data_processed,
            features_mean,
            features_deviation
        ) = prepare_for_training(data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation

        # Initialize model parameters.
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, lambda_param=0):
        (optimized_theta, cost_history) = LogisticRegression.gradient_descent(
            self.data,
            self.labels,
            self.theta,
            lambda_param
        )

        return optimized_theta, cost_history

    @staticmethod
    def gradient_descent(data, labels, initial_theta, lambda_param, max_iteration=500):
        """GRADIENT DESCENT function.

        Iteratively optimizes theta model parameters.

        :param data: the set of training or test data.
        :param labels: training set outputs (0 or 1 that defines the class of an example).
        :param initial_theta: initial model parameters.
        :param lambda_param: regularization parameter.
        :param max_iteration: maximum number of gradient descent steps.
        """

        # Initialize J_history with zeros.
        cost_history = []

        # Calculate the number of features.
        num_features = data.shape[1]

        # Launch gradient descent.
        minification_result = minimize(
            # Function that we're going to minimize.
            lambda current_theta: LogisticRegression.cost_function(
                data, labels, current_theta.reshape((num_features, 1)), lambda_param
            ),
            # Initial values of model parameter.
            initial_theta,
            # We will use conjugate gradient algorithm.
            method='CG',
            # Function that will help to calculate gradient direction on each step.
            jac=lambda current_theta: LogisticRegression.gradient_step(
                data, labels, current_theta.reshape((num_features, 1)), lambda_param
            ),
            # Record gradient descent progress for debugging.
            callback=lambda current_theta: cost_history.append(LogisticRegression.cost_function(
                data, labels, current_theta.reshape((num_features, 1)), lambda_param
            )),
            options={'maxiter': max_iteration}
        )

        # Throw an error in case if gradient descent ended up with error.
        if not minification_result.success:
            raise ArithmeticError('Can not minimize cost function')

        # Reshape the final version of model parameters.
        optimized_theta = minification_result.jac.reshape((num_features, 1))

        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, labels, theta, lambda_param):
        """GRADIENT STEP function.

        It performs one step of gradient descent for theta parameters.

        :param data: the set of training or test data.
        :param labels: training set outputs (0 or 1 that defines the class of an example).
        :param theta: model parameters.
        :param lambda_param: regularization parameter.
        """

        # Initialize number of training examples.
        num_examples = labels.shape[0]

        # Calculate hypothesis predictions and difference with labels.
        predictions = LogisticRegression.hypothesis(data, theta)
        label_diff = predictions - labels

        # Calculate regularization parameter.
        regularization_param = (lambda_param / num_examples) * theta

        # Calculate gradient steps.
        gradients = (1 / num_examples) * (data.T @ label_diff)
        regularized_gradients = gradients + regularization_param

        # We should NOT regularize the parameter theta_zero.
        regularized_gradients[0] = (1 / num_examples) * (data[:, [0]].T @ label_diff)

        return regularized_gradients.T.flatten()

    @staticmethod
    def cost_function(data, labels, theta, lambda_param):
        """Cost function.

        It shows how accurate our model is based on current model parameters.

        :param data: the set of training or test data.
        :param labels: training set outputs (0 or 1 that defines the class of an example).
        :param theta: model parameters.
        :param lambda_param: regularization parameter.
        """

        # Calculate the number of training examples and features.
        num_examples = data.shape[0]

        # Calculate hypothesis.
        predictions = LogisticRegression.hypothesis(data, theta)

        # Calculate regularization parameter
        # Remember that we should not regularize the parameter theta_zero.
        theta_cut = theta[1:, [0]]
        reg_param = (lambda_param / (2 * num_examples)) * (theta_cut.T @ theta_cut)

        # Calculate current predictions cost.
        y_is_set_cost = labels[labels == 1].T @ np.log(predictions[labels == 1])
        y_is_not_set_cost = (1 - labels[labels == 0]).T @ np.log(1 - predictions[labels == 0])
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost) + reg_param

        # Let's extract cost value from the one and only cost numpy matrix cell.
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """Hypothesis function.

        It predicts the output values y based on the input values X and model parameters.

        :param data: data set for what the predictions will be calculated.
        :param theta: model params.
        :return: predictions made by model based on provided theta.
        """

        predictions = sigmoid(data @ theta)

        return predictions
