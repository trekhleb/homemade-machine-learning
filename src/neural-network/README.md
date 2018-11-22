# Neural Network

**Artificial neural networks** (ANN) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains. The neural network itself isn't an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs. Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules.

![Neuron](https://upload.wikimedia.org/wikipedia/commons/1/10/Blausen_0657_MultipolarNeuron.png)

For example, in **image recognition**, they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the results to identify cats in other images. They do this without any prior knowledge about cats, e.g., that they have fur, tails, whiskers and cat-like faces. Instead, they automatically generate identifying characteristics from the learning material that they process.

An ANN is based on a collection of connected units or nodes called **artificial neurons**, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal from one artificial neuron to another. An artificial neuron that receives a signal can process it and then signal additional artificial neurons connected to it.

![Artificial Neuron](https://insights.sei.cmu.edu/sei_blog/sestilli_deeplearning_artificialneuron3.png)

In common ANN implementations, the signal at a connection between artificial neurons is a real number, and the output of each artificial neuron is computed by some non-linear function of the sum of its inputs. The connections between artificial neurons are called **edges**. Artificial neurons and edges typically have a **weight** that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Artificial neurons may have a threshold such that the signal is only sent if the aggregate signal crosses that threshold. Typically, artificial neurons are aggregated into layers. Different layers may perform different kinds of transformations on their inputs. Signals travel from the first layer (the **input layer**), to the last layer (the **output layer**), possibly after traversing the **inner layers** multiple times.

![Neural Network](https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg)

## Neuron Model (Logistic Unit)

Here is a model of one neuron unit.

![neuron](./formulas/neuron.drawio.svg)

![x-0](./formulas/x-0.svg)

![neuron x](./formulas/neuron-x.svg)

Weights:

![neuron weights](./formulas/neuron-weights.svg)

## Network Model (Set of Neurons)

Neural network consists of the neuron units described in the section above.

Let's take a look at simple example model with one hidden layer.

![network model](./formulas/neuron-network.drawio.svg)

![a-i-j](./formulas/a-i-j.svg) - "activation" of unit _i_ in layer _j_.

![Theta-j](./formulas/big-theta-j.svg) - matrix of weights controlling function mapping from layer _j_ to layer _j + 1_. For example for the first layer: ![Theta-1](./formulas/big-theta-1.svg).

![Theta-j](./formulas/L.svg) - total number of layers in network (3 in our example).

![s-l](./formulas/s-l.svg) - number of units (not counting bias unit) in layer _l_.

![K](./formulas/K.svg) - number of output units (1 in our example but could be any real number for multi-class classification).

## Multi-class Classification

In order to make neural network to work with multi-class notification we may use **One-vs-All** approach.

Let's say we want our network to distinguish if there is a _pedestrian_ or _car_ of _motorcycle_ or _truck_ is on the image.

In this case the output layer of our network will have 4 units (input layer will be much bigger and it will have all the pixel from the image. Let's say if all our images will be 20x20 pixels then the input layer will have 400 units each of which will contain the black-white color of the corresponding picture).

![multi-class-network](./formulas/multi-class-network.drawio.svg)

![h-Theta-multi-class](./formulas/multi-class-h.svg)

In this case we would expect our final hypothesis to have following values:

![h-pedestrian](./formulas/h-pedestrian.svg)

![h-car](./formulas/h-car.svg)

![h-motorcycle](./formulas/h-motorcycle.svg)

In this case for the training set:

![training-set](./formulas/training-set.svg)

We would have:

![y-i-multi](./formulas/y-i-multi.svg)

## Forward (or Feedforward) Propagation

Forward propagation is an interactive process of calculating activations for each layer starting from the input layer and going to the output layer.

For the simple network mentioned in a previous section above we're able to calculate activations for second layer based on the input layer and our network parameters:

![a-1-2](./formulas/a-1-2.svg)

![a-2-2](./formulas/a-2-2.svg)

![a-3-2](./formulas/a-3-2.svg)

The output layer activation will be calculated based on the hidden layer activations:

![h-Theta-example](./formulas/h-Theta-example.svg)

Where _g()_ function may be a sigmoid:

![sigmoid](./formulas/sigmoid.svg)

![Sigmoid](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

### Vectorized Implementation of Forward Propagation

Now let's convert previous calculations into more concise vectorized form.

![neuron x](./formulas/neuron-x.svg)

To simplify previous activation equations let's introduce a _z_ variable:

![z-1](./formulas/z-1.svg)

![z-2](./formulas/z-2.svg)

![z-3](./formulas/z-3.svg)

![z-matrix](./formulas/z-matrix.svg)

> Don't forget to add bias units (activations) before propagating to the next layer.
> ![a-bias](./formulas/a-bias.svg)

![z-3-vectorize](./formulas/z-3-vectorized.svg)

![h-Theta-vectorized](./formulas/h-Theta-vectorized.svg)

### Forward Propagation Example

Let's take the following network architecture with 4 layers (input layer, 2 hidden layers and output layer) as an example:

![multi-class-network](./formulas/multi-class-network.drawio.svg)

In this case the forward propagation steps would look like the following:

![forward-propagation-example](./formulas/forward-propagation-example.svg)

## Cost Function

The cost function for the neuron network is quite similar to the logistic regression cost function.

![cost-function](./formulas/cost-function.svg)

![h-Theta](./formulas/h-Theta.svg)

![h-Theta-i](./formulas/h-Theta-i.svg)

## Backpropagation

### Gradient Computation

Backpropagation algorithm has the same purpose as gradient descent for linear or logistic regression - it corrects the values of thetas to minimize a cost function.

In other words we need to be able to calculate partial derivative of cost function for each theta.

![J-partial](./formulas/J-partial.svg)

![multi-class-network](./formulas/multi-class-network.drawio.svg)

Let's assume that:

![delta-j-l](./formulas/delta-j-l.svg) - "error" of node _j_ in layer _l_.

For each output unit (layer _L = 4_):

![delta-4](./formulas/delta-4.svg)

Or in vectorized form:

![delta-4-vectorized](./formulas/delta-4-vectorized.svg)

![delta-3-2](./formulas/delta-3-2.svg)

![sigmoid-gradient](./formulas/sigmoid-gradient.svg) - sigmoid gradient.

![sigmoid-gradient-2](./formulas/sigmoid-gradient-2.svg)

Now we may calculate the gradient step:

![J-partial-detailed](./formulas/J-partial-detailed.svg)

### Backpropagation Algorithm

For training set

![training-set](./formulas/training-set.svg)

We need to set:

![Delta](./formulas/Delta.svg)

![backpropagation](./formulas/backpropagation.svg)

## Random Initialization

Before starting forward propagation we need to initialize Theta parameters. We can not assign zero to all thetas since this would make our network useless because every neuron of the layer will learn the same as its siblings. In other word we need to **break the symmetry**. In order to do so we need to initialize thetas to some small random initial values:

![theta-init](./formulas/theta-init.svg)

## Files

- [demo.m](./demo.m) - demo file that you should run to launch neural network training and to see how the network will recognize handwritten digits.
- [neural_network_train.m](./neural_network_train.m) - function that initialize the neural network and starts its training.
- [neural_network_predict.m](./neural_network_predict.m) - performs prediction of the input data using trained network parameters.
- [debug_initialize_weights.m](./debug_initialize_weights.m) - function that initializes network thetas not randomly for debugging purposes.
- [debug_nn_gradients.m](./debug_nn_gradients.m) - function that helps to debug backpropagation gradients by comparing them to numerically calculated gradients.
- [debug_numerical_gradient.m](./debug_numerical_gradient.m) - calculate the gradient numerically (using small epsilon step at certain point).
- [digits.mat](./digits.mat) - training set of labeled hand-written digits.
- [display_data.m](./display_data.m) - helper function that renders randomly selected digits from training set.
- [fmincg.m](./fmincg.m) - function that does gradient descent (alternative to `fminunc()`).
- [nn_backpropagation.m](./nn_backpropagation.m) - function that performs backpropagation for neural network.
- [nn_cost_function.m](./nn_cost_function.m) - function that calculates neural network cost for specific model parameters.
- [nn_feedforward_propagation.m](./nn_feedforward_propagation.m) - function that performs feedforward propagation for neural network.
- [nn_gradient_step.m](./nn_gradient_step.m) - function that performs one gradient step.
- [nn_params_init.m](./nn_params_init.m) - randomly initializes neural network parameters to brake symmetry.
- [nn_params_roll.m](./nn_params_roll.m) - function that transforms flat vector of thetas into matrices of thetas for each NN layer.
- [sigmoid_gradient.m](./sigmoid_gradient.m) - calculates sigmoid function gradient.
- [sigmoid.m](./sigmoid.m) - sigmoid function.
- [unroll.m](./unroll.m) - transforms matrices of theta for each layer into one flat vector.

### Demo visualizations

![Demo visualization](./formulas/demo.png)

## References

- [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
- [But what is a Neural Network? By 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk)
- [Neural Network on Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [TensorFlow Neural Network Playground](https://playground.tensorflow.org/)
- [Deep Learning by Carnegie Mellon University](https://insights.sei.cmu.edu/sei_blog/2018/02/deep-learning-going-deeper-toward-meaningful-patterns-in-complex-data.html)
