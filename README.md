# Homemade Machine Learning

> The work is in progress yet...

This repository contains examples of popular machine learning algorithms implemented in Python with code examples and mathematics behind them being explained. Each algorithm has Jupyter demo to make the learning process more interactive and to give you a possibility to play with training data, algorithms configurations and to immediately see the results, charts and predictions in your browser. 

The purpose of this repository is _not_ to implement machine learning algorithms using 3<sup>rd</sup> party library "one-liners" _but_ rather to practice and to get better understanding of the mathematics behind each algorithm. That's why all algorithms implementations are "homemade" and not intended to be used for production.   

In most cases the explanations are based on [this great](https://www.coursera.org/learn/machine-learning) machine learning course.

## How to Use This Repository

Depending on how deep you want to dive there are three possible depths:

1. *39 ft:* Browse READMEs and Python files to read about mathematics behind each algorithm and to see examples of how each algorithm may be implemented.
2. *59 ft:* Browse online Jupyter demos and play with each algorithm in your browser by trying to adjust algorithms settings, data sets configuration and to see how the output and predictions vary.
3. *66 ft:* Fork/clone this repo, install all dependencies, launch Jupyter locally and play not only with machine learning algorithms demos but also with theirs Python implementations.

## Supervised Learning

In supervised learning we have a set of training data as an input and a set of labels or "correct answers" for each training set as an output. Then we're training our model (machine learning algorithm parameters) to map the input to the output correctly (to do correct prediction). The ultimate purpose is to find such model parameters that will successfully continue correct input→output mapping (predictions) even for new input examples.

### Regression

In regression problems we do real value predictions. Basically we try to draw a line/plane/n-dimensional plane along the training examples.

_Usage examples: stock price forecast, sales analysis, dependency of any number, etc._

🤖 [Linear Regression](./linear-regression) - example: house prices prediction.

### Classification

In classification problems we split input examples by certain characteristic.

_Usage examples: spam-filters, language detection, finding similar documents, handwritten letters recognition, etc._

🤖 [Logistic Regression](./logistic-regression) - examples: microchip fitness detection, handwritten digits recognitions using one-vs-all approach.

## Unsupervised Learning

Unsupervised learning is a branch of machine learning that learns from test data that has not been labeled, classified or categorized. Instead of responding to feedback, unsupervised learning identifies commonalities in the data and reacts based on the presence or absence of such commonalities in each new piece of data.

### Clustering

In clustering problems we split the training examples by unknown characteristics. The algorithm itself decides what characteristic to use for splitting.

_Usage examples: market segmentation, social networks analysis, organize computing clusters, astronomical data analysis, image compression, etc._

🤖 [K-means algorithm](./k-means) - example: split data into three clusters.

### Anomaly Detection

Anomaly detection (also outlier detection) is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data.

_Usage examples: intrusion detection, fraud detection, system health monitoring, removing anomalous data from the dataset etc._

🤖 [Anomaly Detection using Gaussian distribution](./anomaly-detection) - example: detect overloaded server.

## Neural Network (NN)

The neural network itself isn't an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs.

_Usage examples: as a substitute of all other algorithms in general, image recognition, voice recognition, image processing (applying specific style), language translation, etc._

🤖 [Neural Network: Multilayer Perceptron (MLP)](./neural-network) - example: handwritten digits recognition.

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
