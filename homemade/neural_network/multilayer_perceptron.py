"""Neural Network Module"""

import numpy as np
from ..utils.features import prepare_for_training
from ..utils.hypothesis import sigmoid, sigmoid_gradient


class MultilayerPerceptron:
    """Multilayer Perceptron Class"""

    # pylint: disable=too-many-arguments
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
        self.normalize_data = normalize_data

        # Randomly initialize the weights for each neural network layer.
        self.thetas = MultilayerPerceptron.thetas_init(layers, epsilon)

    def train(self, regularization_param=0, max_iterations=1000, alpha=1):
        """Train the model"""

        # Flatten model thetas for gradient descent.
        unrolled_thetas = MultilayerPerceptron.thetas_unroll(self.thetas)

        # Run gradient descent.
        (optimized_thetas, cost_history) = MultilayerPerceptron.gradient_descent(
            self.data,
            self.labels,
            unrolled_thetas,
            self.layers,
            regularization_param,
            max_iterations,
            alpha
        )

        # Memorize optimized theta parameters.
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_thetas, self.layers)

        return self.thetas, cost_history

    def predict(self, data):
        """Predictions function that does classification using trained model"""

        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]

        num_examples = data_processed.shape[0]

        # Do feedforward propagation with trained neural network params.
        predictions = MultilayerPerceptron.feedforward_propagation(
            data_processed, self.thetas, self.layers
        )

        # Return the index of the output neuron with the highest probability.
        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    @staticmethod
    def gradient_descent(
            data, labels, unrolled_theta, layers, regularization_param, max_iteration, alpha
    ):
        # pylint: disable=too-many-arguments
        """Gradient descent function.

        Iteratively optimizes theta model parameters.

        :param data: the set of training or test data.
        :param labels: training set outputs (0 or 1 that defines the class of an example).
        :param unrolled_theta: initial model parameters.
        :param layers: model layers configuration.
        :param regularization_param: regularization parameter.
        :param max_iteration: maximum number of gradient descent steps.
        :param alpha: gradient descent step size.
        """

        optimized_theta = unrolled_theta

        # Initialize cost history list.
        cost_history = []

        for _ in range(max_iteration):
            # Get current cost.
            cost = MultilayerPerceptron.cost_function(
                data,
                labels,
                MultilayerPerceptron.thetas_roll(optimized_theta, layers),
                layers,
                regularization_param
            )

            # Save current cost value to build plots later.
            cost_history.append(cost)

            # Get the next gradient step directions.
            theta_gradient = MultilayerPerceptron.gradient_step(
                data, labels, optimized_theta, layers, regularization_param
            )

            # Adjust theta values according to the next gradient step.
            optimized_theta = optimized_theta - alpha * theta_gradient

        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, labels, unrolled_thetas, layers, regularization_param):
        """Gradient step function.

        Computes the cost and gradient of the neural network for unrolled theta parameters.

        :param data: training set.
        :param labels: training set labels.
        :param unrolled_thetas: model parameters.
        :param layers: model layers configuration.
        :param regularization_param: parameters that fights with model over-fitting.
        """

        # Reshape nn_params back into the matrix parameters.
        thetas = MultilayerPerceptron.thetas_roll(unrolled_thetas, layers)

        # Do backpropagation.
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(
            data, labels, thetas, layers, regularization_param
        )

        # Unroll thetas gradients.
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)

        return thetas_unrolled_gradients

    # pylint: disable=R0914
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
        """Feedforward propagation function"""

        # Calculate the total number of layers.
        num_layers = len(layers)

        # Calculate the number of training examples.
        num_examples = data.shape[0]

        # Input layer (l=1)
        in_layer_activation = data

        # Propagate to hidden layers.
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            out_layer_activation = sigmoid(in_layer_activation @ theta.T)
            # Add bias units.
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation

        # Output layer should not contain bias units.
        return in_layer_activation[:, 1:]

    # pylint: disable=R0914
    @staticmethod
    def back_propagation(data, labels, thetas, layers, regularization_param):
        """Backpropagation function"""

        # Get total number of layers.
        num_layers = len(layers)

        # Get total number of training examples and features.
        (num_examples, num_features) = data.shape

        # Get the number of possible output labels.
        num_label_types = layers[-1]

        # Initialize big delta - aggregated delta values for all training examples that will
        # indicate how exact theta need to be changed.
        deltas = {}
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            deltas[layer_index] = np.zeros((out_count, in_count + 1))

        # Let's go through all examples.
        for example_index in range(num_examples):
            # We will store layers inputs and activations in order to re-use it later.
            layers_inputs = {}
            layers_activations = {}

            # Setup input layer activations.
            layer_activation = data[example_index, :].reshape((num_features, 1))
            layers_activations[0] = layer_activation

            # Perform a feedforward pass for current training example.
            for layer_index in range(num_layers - 1):
                layer_theta = thetas[layer_index]
                layer_input = layer_theta @ layer_activation
                layer_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))

                layers_inputs[layer_index + 1] = layer_input
                layers_activations[layer_index + 1] = layer_activation

            # Remove bias units from the output activations.
            output_layer_activation = layer_activation[1:, :]

            # Calculate deltas.

            # For input layer we don't calculate delta because we do not
            # associate error with the input.
            delta = {}

            # Convert the output from number to vector (i.e. 5 to [0; 0; 0; 0; 1; 0; 0; 0; 0; 0])
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1

            # Calculate deltas for the output layer for current training example.
            delta[num_layers - 1] = output_layer_activation - bitwise_label

            # Calculate small deltas for hidden layers for current training example.
            # The loops should go for the layers L, L-1, ..., 1.
            for layer_index in range(num_layers - 2, 0, -1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index + 1]
                layer_input = layers_inputs[layer_index]

                # Add bias row to the layer input.
                layer_input = np.vstack((np.array([[1]]), layer_input))

                # Calculate row delta and take off the bias row from it.
                delta[layer_index] = (layer_theta.T @ next_delta) * sigmoid_gradient(layer_input)
                delta[layer_index] = delta[layer_index][1:, :]

            # Accumulate the gradient (update big deltas).
            for layer_index in range(num_layers - 1):
                layer_delta = delta[layer_index + 1] @ layers_activations[layer_index].T
                deltas[layer_index] = deltas[layer_index] + layer_delta

        # Obtain un-regularized gradient for the neural network cost function.
        for layer_index in range(num_layers - 1):
            # Remember that we should NOT be regularizing the first column of theta.
            current_delta = deltas[layer_index]
            current_delta = np.hstack((np.zeros((current_delta.shape[0], 1)), current_delta[:, 1:]))

            # Calculate regularization.
            regularization = (regularization_param / num_examples) * current_delta

            # Regularize deltas.
            deltas[layer_index] = (1 / num_examples) * deltas[layer_index] + regularization

        return deltas

    @staticmethod
    def thetas_init(layers, epsilon):
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
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 2 * epsilon - epsilon

        return thetas

    @staticmethod
    def thetas_unroll(thetas):
        """Unrolls cells of theta matrices into one long vector."""

        unrolled_thetas = np.array([])
        num_theta_layers = len(thetas)
        for theta_layer_index in range(num_theta_layers):
            # Unroll cells into vector form.
            unrolled_thetas = np.hstack((unrolled_thetas, thetas[theta_layer_index].flatten()))

        return unrolled_thetas

    @staticmethod
    def thetas_roll(unrolled_thetas, layers):
        """Rolls NN params vector into the matrix"""

        # Get total numbers of layers.
        num_layers = len(layers)

        # Init rolled thetas dictionary.
        thetas = {}
        unrolled_shift = 0

        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]

            thetas_width = in_count + 1  # We need to remember about bias unit.
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height

            # We need to remember about bias units when rolling up params.
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_thetas_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_thetas_unrolled.reshape((thetas_height, thetas_width))

            # Shift frame to the right.
            unrolled_shift = unrolled_shift + thetas_volume

        return thetas
