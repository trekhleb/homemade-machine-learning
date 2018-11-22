# Homemade Machine Learning

> This repository contains examples of popular machine learning algorithms implemented in Python with code examples and mathematics behind them being explained. Each algorithm has Jupyter demo to make the learning process more interactive and to give you a possibility to play with training data, algorithms configurations and to immediately see the results, charts and predictions in your browser. 

The purpose of this repository is _not_ to implement machine learning algorithms using 3<sup>rd</sup> party library "one-liners" _but_ rather to practice and to get better understanding of the mathematics behind each algorithm. That's why all algorithms implementations are "homemade" and not intended to be used for production. In most cases the explanations are based on [this great](https://www.coursera.org/learn/machine-learning) machine learning course.

#### How to Use This Repository

1. **Observe** READMEs and Python files to read about mathematics behind each algorithm and to see examples of how each algorithm may be implemented.
2. **Interact** with online Jupyter demos and play with each algorithm in your browser by trying to adjust algorithms settings, data sets configuration and to see how the output and predictions vary.
3. **Change** the way the algorithms are implemented by forking/cloning this repo, installing all dependencies, launching Jupyter locally and playing with Python implementations.

## Supervised Learning

In supervised learning we have a set of training data as an input and a set of labels or "correct answers" for each training set as an output. Then we're training our model (machine learning algorithm parameters) to map the input to the output correctly (to do correct prediction). The ultimate purpose is to find such model parameters that will successfully continue correct input→output mapping (predictions) even for new input examples.

### Regression

In regression problems we do real value predictions. Basically we try to draw a line/plane/n-dimensional plane along the training examples.

_Usage examples: stock price forecast, sales analysis, dependency of any number, etc._

#### 🤖 Linear Regression

- [Linear Regression (Math)](./src/linear_regression)
- Jupyter Demos:
    - [Univariate Linear Regression Demo](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/src/linear_regression/univariate_linear_regression_demo.ipynb) - predict country happiness score by economy GDP per capita.
    - Multivariate Linear Regression Demo - ⏳ in progress...
    - Polynomial Regression Demo - ⏳ in progress...

### Classification

In classification problems we split input examples by certain characteristic.

_Usage examples: spam-filters, language detection, finding similar documents, handwritten letters recognition, etc._

#### 🤖 Logistic Regression

- [Logistic Regression (Math)](./src/logistic_regression)
- Jupyter Demos:
    - ⏳ in progress...

## Unsupervised Learning

Unsupervised learning is a branch of machine learning that learns from test data that has not been labeled, classified or categorized. Instead of responding to feedback, unsupervised learning identifies commonalities in the data and reacts based on the presence or absence of such commonalities in each new piece of data.

### Clustering

In clustering problems we split the training examples by unknown characteristics. The algorithm itself decides what characteristic to use for splitting.

_Usage examples: market segmentation, social networks analysis, organize computing clusters, astronomical data analysis, image compression, etc._

#### 🤖 K-means Algorithm

- [K-means Algorithm (Math)](./src/k_means)
- Jupyter Demos:
    - ⏳ in progress...

### Anomaly Detection

Anomaly detection (also outlier detection) is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data.

_Usage examples: intrusion detection, fraud detection, system health monitoring, removing anomalous data from the dataset etc._

#### 🤖 Anomaly Detection using Gaussian Distribution

- [Anomaly Detection using Gaussian Distribution (Math)](./src/anomaly_detection)
- Jupyter Demos:
    - ⏳ in progress...

## Neural Network (NN)

The neural network itself isn't an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs.

_Usage examples: as a substitute of all other algorithms in general, image recognition, voice recognition, image processing (applying specific style), language translation, etc._

#### 🤖 Multilayer Perceptron (MLP)

- [Multilayer Perceptron (Math)](./src/neural_network)
- Jupyter Demos:
    - ⏳ in progress...

## Machine Learning Map

![Machine Learning Map](./images/machine-learning-map.png)

The source of the following machine learning topics map is [this wonderful blog post](https://vas3k.ru/blog/machine_learning/)

## Prerequisites

**Installing Python**

Make sure that you have [Python3 installed](https://realpython.com/installing-python/) on your machine.

You might want to use [venv](https://docs.python.org/3/library/venv.html) standard Python library
to create virtual environments and have Python, pip and all dependent packages to be installed and 
served from the local project directory to avoid messing with system wide packages and their 
versions.

Depending on your installation you might have access to Python3 interpreter either by
running `python` or `python3`. The same goes for pip package manager - it may be accessible either
by running `pip` or `pip3`.

You may check your Python version by running:

```bash
python --version
```

Note that in this repository whenever you see `python` it will be assumed that it is Python **3**.

**Installing dependencies**

Install all dependencies that are required for the project by running:

```bash
pip install -r requirements.txt
```
