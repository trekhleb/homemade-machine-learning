import numpy as np
from scipy.optimize import minimize
from ..utils.features import prepare_for_training
from ..utils.hypothesis import sigmoid


class MultilayerPerceptron:
    """Multilayer Perceptron Class"""

    def __init__(self, data, labels, layers, epsilon, normalize_data=False):
        """Multilayer perceptron constructor.

        :param data: training set.
        :param labels: training set outputs (correct values).
        :param layers: network layers configuration.
        :param epsilon: Defines the range for initial theta values.
        :param normalize_data: flag that indicates that features should be normalized.
        """

        # Normalize features and add ones column.
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]

        self.data = data_processed
        self.labels = labels
        self.layers = layers
        self.epsilon = epsilon

        # Randomly initialize the weights for each neural network layer.
        self.thetas = MultilayerPerceptron.init_layers_thetas(layers, epsilon)

    def train(self, regularization_param=0, max_iterations=1000):
        # Flatten model thetas for gradient descent.
        unrolled_thetas = MultilayerPerceptron.unroll_thetas(self.thetas)

        # Init cost history array.
        cost_histories = []

        initial_cost = MultilayerPerceptron.cost_function(
            self.data,
            self.labels,
            self.thetas,
            self.layers,
            regularization_param
        )

        print(initial_cost)

        # Run gradient descent.
        # (current_theta, cost_history) = MultilayerPerceptron.gradient_descent(
        #     self.data,
        #     current_labels,
        #     unrolled_thetas,
        #     regularization_param,
        #     max_iterations,
        # )

        return self.thetas, cost_histories

    # @staticmethod
    # def gradient_descent(data, labels, initial_theta, lambda_param, max_iteration):
    #     """Gradient descent function.
    #
    #     Iteratively optimizes theta model parameters.
    #
    #     :param data: the set of training or test data.
    #     :param labels: training set outputs (0 or 1 that defines the class of an example).
    #     :param initial_theta: initial model parameters.
    #     :param lambda_param: regularization parameter.
    #     :param max_iteration: maximum number of gradient descent steps.
    #     """
    #
    #     # Initialize cost history list.
    #     cost_history = []
    #
    #     # Launch gradient descent.
    #     minification_result = minimize(
    #         # Function that we're going to minimize.
    #         lambda current_theta: MultilayerPerceptron.cost_function(
    #             data, labels, current_theta.reshape((num_features, 1)), lambda_param
    #         ),
    #         # Initial values of model parameter.
    #         initial_theta,
    #         # We will use conjugate gradient algorithm.
    #         method='CG',
    #         # Function that will help to calculate gradient direction on each step.
    #         jac=lambda current_theta: MultilayerPerceptron.gradient_step(
    #             data, labels, current_theta.reshape((num_features, 1)), lambda_param
    #         ),
    #         # Record gradient descent progress for debugging.
    #         callback=lambda current_theta: cost_history.append(MultilayerPerceptron.cost_function(
    #             data, labels, current_theta.reshape((num_features, 1)), lambda_param
    #         )),
    #         options={'maxiter': max_iteration}
    #     )
    #
    #     # Throw an error in case if gradient descent ended up with error.
    #     if not minification_result.success:
    #         raise ArithmeticError('Can not minimize cost function: ' + minification_result.message)
    #
    #     # Reshape the final version of model parameters.
    #     optimized_theta = minification_result.x.reshape((num_features, 1))
    #
    #     return optimized_theta, cost_history

    @staticmethod
    def cost_function(data, labels, thetas, layers, regularization_param):
        """Cost function.

        It shows how accurate our model is based on current model parameters.

        :param data: the set of training or test data.
        :param labels: training set outputs (0 or 1 that defines the class of an example).
        :param thetas: model parameters.
        :param layers: layers configuration.
        :param regularization_param: regularization parameter.
        """

        # Get total number of layers.
        num_layers = len(layers)

        # Get total number of training examples.
        num_examples = data.shape[0]

        # Get the size of output layer (number of labels).
        num_labels = layers[-1]

        # Feedforward the neural network.
        predictions = MultilayerPerceptron.feedforward_propagation(data, thetas, layers)

        # Compute the cost.

        # For now the labels vector is just an expected number for each input example.
        # We need to convert every result from number to vector that will illustrate
        # the output we're expecting. For example instead of having just number 5
        # we want to expect [0 0 0 0 1 0 0 0 0 0]. The bit is set for 5th position.
        bitwise_labels = np.zeros((num_examples, num_labels))
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1

        # Calculate regularization parameter.
        theta_square_sum = 0
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            # Don't try to regularize bias thetas.
            theta_square_sum = theta_square_sum + np.sum(theta[:, 1:] ** 2)

        regularization = (regularization_param / (2 * num_examples)) * theta_square_sum

        # Calculate the cost with regularization.
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
        cost = (-1 / num_examples) * (bit_set_cost + bit_not_set_cost) + regularization

        return cost

    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        # Calculate the total number of layers.
        num_layers = len(layers)

        # Calculate the number of training examples.
        num_examples = data.shape[0]

        # Input layer (l=1)
        layer_in = data

        # Propagate to hidden layers.
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            layer_out = sigmoid(layer_in @ theta.T)
            # Add bias units.
            layer_out = np.hstack((np.ones((num_examples, 1)), layer_out))
            layer_in = layer_out

        # Output layer should not contain bias units.
        return layer_in[:, 1:]

    @staticmethod
    def init_layers_thetas(layers, epsilon):
        """Randomly initialize the weights for each neural network layer

        Each layer will have its own theta matrix W with L_in incoming connections and L_out
        outgoing connections. Note that W will be set to a matrix of size(L_out, 1 + L_in) as the
        first column of W handles the "bias" terms.

        :param layers:
        :param epsilon:
        :return:
        """

        # Get total number of layers.
        num_layers = len(layers)

        # Generate initial thetas for each layer.
        thetas = {}

        # Generate Thetas only for input and hidden layers.
        # There is no need to generate Thetas for the output layer.
        for layer_index in range(num_layers - 1):
            layers_in = layers[layer_index]
            layers_out = layers[layer_index + 1]
            thetas[layer_index] = np.random.rand(layers_out, layers_in + 1) * 2 * epsilon - epsilon

        return thetas

    @staticmethod
    def unroll_thetas(thetas):
        """Unrolls cells of theta matrices into one long vector."""

        unrolled_thetas = []
        num_theta_layers = len(thetas)
        for theta_layer_index in range(num_theta_layers):
            # Unroll cells into vector form.
            unrolled_thetas.extend(thetas[theta_layer_index].flatten())

        return unrolled_thetas
