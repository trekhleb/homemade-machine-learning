# Datasets

This is a list of datasets that are used for Jupyter Notebook demos in this repository.

### MNIST (Handwritten Digits)

> [mnist-demo.csv](mnist-demo.csv)

_Source: [Kaggle](https://www.kaggle.com/oddrationale/mnist-in-csv/home)_

A sample of original MNIST dataset in a CSV format. Instead of using full dataset with 60000 training examples the dataset consists of just 10000 examples.

Each row in the dataset consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values (28x28 pixels image) are the pixel values (a number from 0 to 255).

### Fashion MNIST

> [fashion-mnist-demo.csv](fashion-mnist-demo.csv)

_Source: [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)_

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.
 
 Instead of using full dataset with 60000 training examples we will use cut dataset of just 5000 examples that we will also split into training and testing sets.

### World Happiness Report 2017

> [world-happiness-report-2017.csv](world-happiness-report-2017.csv)

_Source: [Kaggle](https://www.kaggle.com/unsdsn/world-happiness#2017.csv)_

Happiness rank and scores by country, 2017.

### Iris Flowers

> [iris.csv](iris.csv)

_Source: [ics.uci.edu](http://archive.ics.uci.edu/ml/datasets/Iris)_

Iris data set data set consists of several samples from each of three species of Iris (`Iris setosa`, `Iris virginica` and `Iris versicolor`). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

### Microchips Tests (Artificial)

> [microchips-tests.csv](microchips-tests.csv)

_Source: [Machine Learning at Coursera](https://www.coursera.org/learn/machine-learning)_

Artificial dataset in which `param_1` and `param_2` produce non-linear decision boundary.

### Non-Linear Y(X) Dependency (Artificial)

> [non-linear-regression-x-y.csv](non-linear-regression-x-y.csv)

_Source: [Machine Learning at Coursera](https://www.coursera.org/learn/machine-learning)_

Artificial dataset that contains non-linear y(x) dependency.

### Server Operational Parameters

> [server-operational-params.csv](server-operational-params.csv)

_Source: [Machine Learning at Coursera](https://www.coursera.org/learn/machine-learning)_

Dataset of server operational parameters containing the `Latency(Throughput)` dependency. 
