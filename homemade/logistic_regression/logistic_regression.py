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
        LogisticRegression.gradient_descent(
            self.data,
            self.labels,
            self.theta,
            lambda_param
        )

        # def rosen(x):
        #     """The Rosenbrock function"""
        #     return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
        #
        # def rosen_der(x):
        #     xm = x[1:-1]
        #     xm_m1 = x[:-2]
        #     xm_p1 = x[2:]
        #     der = np.zeros_like(x)
        #     der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
        #     der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        #     der[-1] = 200 * (x[-1] - x[-2] ** 2)
        #     return der
        #
        # x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        #
        # res = minimize(
        #     rosen,
        #     x0,
        #     method='CG',
        #     jac=rosen_der,
        #     options={
        #         'maxiter': 500
        #     }
        # )
        #
        # print(res.x)
        # print(res.fun)
        # print(res.jac)
        # print(res.success)

        # cost_history = self.gradient_descent(lambda_param)

        # return self.theta, cost_history
        pass

    @staticmethod
    def gradient_descent(data, labels, initial_theta, lambda_param, max_iteration=500):
        # Initialize J_history with zeros.
        cost_history = []

        # print(initial_theta[1:, [0]])

        # print(LogisticRegression.cost_function(
        #     data,
        #     labels,
        #     initial_theta,
        #     lambda_param
        # ))
        #
        # print(LogisticRegression.gradient_step(
        #     data,
        #     labels,
        #     initial_theta,
        #     lambda_param
        # ))

        # num_features = data.shape[1]
        #
        # minification_result = minimize(
        #     lambda current_theta: LogisticRegression.cost_function(
        #         data, labels, current_theta.reshape((num_features, 1)), lambda_param
        #     ),
        #     initial_theta,
        #     method='CG',
        #     jac=lambda current_theta: LogisticRegression.gradient_step(
        #         data, labels, current_theta.reshape((num_features, 1)), lambda_param
        #     ),
        #     options={'maxiter': max_iteration}
        # )
        #
        # print(minification_result)

        return cost_history

    @staticmethod
    def gradient_step(data, labels, theta, lambda_param):
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

        return regularized_gradients

    @staticmethod
    def cost_function(data, labels, theta, lambda_param):
        """Cost function.

        It shows how accurate our model is based on current model parameters.

        :param data: the set of training or test data.
        :param labels: training set outputs (0 or 1 that defines the class of an example).
        :param theta: model parameters.
        :param lambda_param: regularization parameter
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
        y_is_set_cost = labels.T @ np.log(predictions)
        y_is_not_set_cost = (1 - labels).T @ np.log(1 - predictions)
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

        # Get number of examples.
        num_examples = data.shape[0]

        predictions = sigmoid(data @ theta)

        return predictions.reshape((num_examples, 1))
