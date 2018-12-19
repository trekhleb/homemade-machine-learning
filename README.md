# Homemade Machine Learning

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/trekhleb/homemade-machine-learning/master?filepath=notebooks)

_For Octave/MatLab version of this repository please check [machine-learning-octave](https://github.com/trekhleb/machine-learning-octave) project._

> This repository contains examples of popular machine learning algorithms implemented in **Python** with mathematics behind them being explained. Each algorithm has interactive **Jupyter Notebook** demo that allows you to play with training data, algorithms configurations and to immediately see the results, charts and predictions in your browser. In most cases the explanations are based on great [machine learning course](https://www.coursera.org/learn/machine-learning) by Andrew Ng.

The purpose of this repository is _not_ to implement machine learning algorithms by using 3<sup>rd</sup> party library "one-liners" _but_ rather to practice and to get better understanding of the mathematics behind each algorithm. That's why all algorithms implementations are called "homemade" and not intended to be used for production.

## Supervised Learning

In supervised learning we have a set of training data as an input and a set of labels or "correct answers" for each training set as an output. Then we're training our model (machine learning algorithm parameters) to map the input to the output correctly (to do correct prediction). The ultimate purpose is to find such model parameters that will successfully continue correct input→output mapping (predictions) even for new input examples.

### Regression

In regression problems we do real value predictions. Basically we try to draw a line/plane/n-dimensional plane along the training examples.

_Usage examples: stock price forecast, sales analysis, dependency of any number, etc._

#### 🤖 Linear Regression

- 📗 [Math | Linear Regression](homemade/linear_regression) - theory and links for further readings
- ⚙️ [Code | Linear Regression](homemade/linear_regression/linear_regression.py) - implementation example
- ▶️ [Demo | Univariate Linear Regression](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/univariate_linear_regression_demo.ipynb) - predict `country happiness` score by `economy GDP`
- ▶️ [Demo | Multivariate Linear Regression](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/multivariate_linear_regression_demo.ipynb) - predict `country happiness` score by `economy GDP` and `freedom index`
- ▶️ [Demo | Non-linear Regression](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/linear_regression/non_linear_regression_demo.ipynb) - use linear regression with _polynomial_ and _sinusoid_ features to predict non-linear dependencies.

### Classification

In classification problems we split input examples by certain characteristic.

_Usage examples: spam-filters, language detection, finding similar documents, handwritten letters recognition, etc._

#### 🤖 Logistic Regression

- 📗 [Math | Logistic Regression](homemade/logistic_regression) - theory and links for further readings
- ⚙️ [Code | Logistic Regression](homemade/logistic_regression/logistic_regression.py) - implementation example
- ▶️ [Demo | Logistic Regression With Linear Boundary](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/logistic_regression_with_linear_boundary_demo.ipynb) - predict Iris flower `class` based on `petal_length` and `petal_width`
- ▶️ [Demo | Logistic Regression With Non-Linear Boundary](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/logistic_regression_with_non_linear_boundary_demo.ipynb) - predict microchip `validity` based on `param_1` and `param_2`

## Unsupervised Learning

Unsupervised learning is a branch of machine learning that learns from test data that has not been labeled, classified or categorized. Instead of responding to feedback, unsupervised learning identifies commonalities in the data and reacts based on the presence or absence of such commonalities in each new piece of data.

### Clustering

In clustering problems we split the training examples by unknown characteristics. The algorithm itself decides what characteristic to use for splitting.

_Usage examples: market segmentation, social networks analysis, organize computing clusters, astronomical data analysis, image compression, etc._

#### 🤖 K-means Algorithm

- 📗 [Math | K-means Algorithm](homemade/k_means)
- ⚙️ Code | Linear Regression
- ▶️ Demo | K-means Algorithm

### Anomaly Detection

Anomaly detection (also outlier detection) is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data.

_Usage examples: intrusion detection, fraud detection, system health monitoring, removing anomalous data from the dataset etc._

#### 🤖 Anomaly Detection using Gaussian Distribution

- 📗 [Math | Anomaly Detection using Gaussian Distribution](homemade/anomaly_detection)
- ⚙️ Code | Anomaly Detection using Gaussian Distribution
- ▶️ Demo | Anomaly Detection

## Neural Network (NN)

The neural network itself isn't an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs.

_Usage examples: as a substitute of all other algorithms in general, image recognition, voice recognition, image processing (applying specific style), language translation, etc._

#### 🤖 Multilayer Perceptron (MLP)

- 📗 [Math | Multilayer Perceptron](homemade/neural_network)
- ⚙️ Code | Multilayer Perceptron
- ▶️ Demo | Multilayer Perceptron

## Machine Learning Map

![Machine Learning Map](images/machine-learning-map.png)

The source of the following machine learning topics map is [this wonderful blog post](https://vas3k.ru/blog/machine_learning/)

## Prerequisites

#### Installing Python

Make sure that you have [Python installed](https://realpython.com/installing-python/) on your machine.

You might want to use [venv](https://docs.python.org/3/library/venv.html) standard Python library
to create virtual environments and have Python, `pip` and all dependent packages to be installed and 
served from the local project directory to avoid messing with system wide packages and their 
versions.

#### Installing Dependencies

Install all dependencies that are required for the project by running:

```bash
pip install -r requirements.txt
```

#### Launching Jupyter Locally

All demos in the project may be run directly in your browser without installing Jupyter locally. But if you want to launch [Jupyter Notebook](http://jupyter.org/) locally you may do it be running the following command from the root folder of the project:

```bash
jupyter notebook
```
After this Jupyter Notebook will be accessible by `http://localhost:8888`.
