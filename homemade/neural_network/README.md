# Neural Network

## Jupyter Demos

▶️ [Demo | Multilayer Perceptron | MNIST](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/neural_network/multilayer_perceptron_demo.ipynb) - recognize handwritten digits from `28x28` pixel images.

▶️ [Demo | Multilayer Perceptron | Fashion MNIST](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/neural_network/multilayer_perceptron_fashion_demo.ipynb) - recognize the type of clothes (Dress, Coat, Sandal, etc.) from `28x28` pixel images.

## Definition

**Artificial neural networks** (ANN) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains. The neural network itself isn't an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs. Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules.

![Neuron](https://upload.wikimedia.org/wikipedia/commons/1/10/Blausen_0657_MultipolarNeuron.png)

For example, in **image recognition**, they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the results to identify cats in other images. They do this without any prior knowledge about cats, e.g., that they have fur, tails, whiskers and cat-like faces. Instead, they automatically generate identifying characteristics from the learning material that they process.

An ANN is based on a collection of connected units or nodes called **artificial neurons**, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal from one artificial neuron to another. An artificial neuron that receives a signal can process it and then signal additional artificial neurons connected to it.

![Artificial Neuron](https://insights.sei.cmu.edu/sei_blog/sestilli_deeplearning_artificialneuron3.png)

In common ANN implementations, the signal at a connection between artificial neurons is a real number, and the output of each artificial neuron is computed by some non-linear function of the sum of its inputs. The connections between artificial neurons are called **edges**. Artificial neurons and edges typically have a **weight** that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Artificial neurons may have a threshold such that the signal is only sent if the aggregate signal crosses that threshold. Typically, artificial neurons are aggregated into layers. Different layers may perform different kinds of transformations on their inputs. Signals travel from the first layer (the **input layer**), to the last layer (the **output layer**), possibly after traversing the **inner layers** multiple times.

![Neural Network](https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg)

A **multilayer perceptron (MLP)** is a class of feedforward artificial neural network. An MLP consists of, at least, three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

## Neuron Model (Logistic Unit)

Here is a model of one neuron unit.

![neuron](../../images/neural_network/neuron.drawio.svg)

![x-0](../../images/neural_network/x-0.svg)

![neuron x](../../images/neural_network/neuron-x.svg)

Weights:

![neuron weights](../../images/neural_network/neuron-weights.svg)

## Network Model (Set of Neurons)

Neural network consists of the neuron units described in the section above.

Let's take a look at simple example model with one hidden layer.

![network model](../../images/neural_network/neuron-network.drawio.svg)

![a-i-j](../../images/neural_network/a-i-j.svg) - "activation" of unit _i_ in layer _j_.

![Theta-j](../../images/neural_network/big-theta-j.svg) - matrix of weights controlling function mapping from layer _j_ to layer _j + 1_. For example for the first layer: ![Theta-1](../../images/neural_network/big-theta-1.svg).

![Theta-j](../../images/neural_network/L.svg) - total number of layers in network (3 in our example).

![s-l](../../images/neural_network/s-l.svg) - number of units (not counting bias unit) in layer _l_.

![K](../../images/neural_network/K.svg) - number of output units (1 in our example but could be any real number for multi-class classification).

## Multi-class Classification

In order to make neural network to work with multi-class notification we may use **One-vs-All** approach.

Let's say we want our network to distinguish if there is a _pedestrian_ or _car_ of _motorcycle_ or _truck_ is on the image.

In this case the output layer of our network will have 4 units (input layer will be much bigger and it will have all the pixel from the image. Let's say if all our images will be 20x20 pixels then the input layer will have 400 units each of which will contain the black-white color of the corresponding picture).

![multi-class-network](../../images/neural_network/multi-class-network.drawio.svg)

![h-Theta-multi-class](../../images/neural_network/multi-class-h.svg)

In this case we would expect our final hypothesis to have following values:

![h-pedestrian](../../images/neural_network/h-pedestrian.svg)

![h-car](../../images/neural_network/h-car.svg)

![h-motorcycle](../../images/neural_network/h-motorcycle.svg)

In this case for the training set:

![training-set](../../images/neural_network/training-set.svg)

We would have:

![y-i-multi](../../images/neural_network/y-i-multi.svg)

## Forward (or Feedforward) Propagation

Forward propagation is an interactive process of calculating activations for each layer starting from the input layer and going to the output layer.

For the simple network mentioned in a previous section above we're able to calculate activations for second layer based on the input layer and our network parameters:

![a-1-2](../../images/neural_network/a-1-2.svg)

![a-2-2](../../images/neural_network/a-2-2.svg)

![a-3-2](../../images/neural_network/a-3-2.svg)

The output layer activation will be calculated based on the hidden layer activations:

![h-Theta-example](../../images/neural_network/h-Theta-example.svg)

Where _g()_ function may be a sigmoid:

![sigmoid](../../images/neural_network/sigmoid.svg)

![Sigmoid](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

### Vectorized Implementation of Forward Propagation

Now let's convert previous calculations into more concise vectorized form.

![neuron x](../../images/neural_network/neuron-x.svg)

To simplify previous activation equations let's introduce a _z_ variable:

![z-1](../../images/neural_network/z-1.svg)

![z-2](../../images/neural_network/z-2.svg)

![z-3](../../images/neural_network/z-3.svg)

![z-matrix](../../images/neural_network/z-matrix.svg)

> Don't forget to add bias units (activations) before propagating to the next layer.
> ![a-bias](../../images/neural_network/a-bias.svg)

![z-3-vectorize](../../images/neural_network/z-3-vectorized.svg)

![h-Theta-vectorized](../../images/neural_network/h-Theta-vectorized.svg)

### Forward Propagation Example

Let's take the following network architecture with 4 layers (input layer, 2 hidden layers and output layer) as an example:

![multi-class-network](../../images/neural_network/multi-class-network.drawio.svg)

In this case the forward propagation steps would look like the following:

![forward-propagation-example](../../images/neural_network/forward-propagation-example.svg)

## Cost Function

The cost function for the neuron network is quite similar to the logistic regression cost function.

![cost-function](../../images/neural_network/cost-function.svg)

![h-Theta](../../images/neural_network/h-Theta.svg)

![h-Theta-i](../../images/neural_network/h-Theta-i.svg)

## Backpropagation

### Gradient Computation

Backpropagation algorithm has the same purpose as gradient descent for linear or logistic regression - it corrects the values of thetas to minimize a cost function.

In other words we need to be able to calculate partial derivative of cost function for each theta.

![J-partial](../../images/neural_network/J-partial.svg)

![multi-class-network](../../images/neural_network/multi-class-network.drawio.svg)

Let's assume that:

![delta-j-l](../../images/neural_network/delta-j-l.svg) - "error" of node _j_ in layer _l_.

For each output unit (layer _L = 4_):

![delta-4](../../images/neural_network/delta-4.svg)

Or in vectorized form:

![delta-4-vectorized](../../images/neural_network/delta-4-vectorized.svg)

![delta-3-2](../../images/neural_network/delta-3-2.svg)

![sigmoid-gradient](../../images/neural_network/sigmoid-gradient.svg) - sigmoid gradient.

![sigmoid-gradient-2](../../images/neural_network/sigmoid-gradient-2.svg)

Now we may calculate the gradient step:

![J-partial-detailed](../../images/neural_network/J-partial-detailed.svg)

### Backpropagation Algorithm

For training set

![training-set](../../images/neural_network/training-set.svg)

We need to set:

![Delta](../../images/neural_network/Delta.svg)

![backpropagation](../../images/neural_network/backpropagation.svg)

## Random Initialization

Before starting forward propagation we need to initialize Theta parameters. We can not assign zero to all thetas since this would make our network useless because every neuron of the layer will learn the same as its siblings. In other word we need to **break the symmetry**. In order to do so we need to initialize thetas to some small random initial values:

![theta-init](../../images/neural_network/theta-init.svg)

## References

- [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
- [But what is a Neural Network? By 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk)
- [Neural Network on Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [TensorFlow Neural Network Playground](https://playground.tensorflow.org/)
- [Deep Learning by Carnegie Mellon University](https://insights.sei.cmu.edu/sei_blog/2018/02/deep-learning-going-deeper-toward-meaningful-patterns-in-complex-data.html)
