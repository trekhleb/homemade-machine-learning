# Anomaly Detection Using Gaussian Distribution

## Gaussian (Normal) Distribution

The **normal** (or **Gaussian**) **distribution** is a very common continuous probability distribution. Normal distributions are important in statistics and are often used in the natural and social sciences to represent real-valued random variables whose distributions are not known. A random variable with a Gaussian distribution is said to be normally distributed and is called a normal deviate.

Let's say:

![x-in-R](./formulas/x-in-R.svg)

If _x_ is normally distributed then it may be displayed as follows.

![Gaussian Distribution](https://upload.wikimedia.org/wikipedia/commons/7/74/Normal_Distribution_PDF.svg)

![mu](./formulas/mu.svg) - mean value,

![sigma-2](./formulas/sigma-2.svg) - variance.

![x-normal](./formulas/x-normal.svg) - "~" means that _"x is distibuted as ..."_

Then Gaussian distribution (probability that some _x_ may be a part of distribution with certain mean and variance) is given by:

![Gaussian Distribution](./formulas/p.svg)

## Estimating Parameters for a Gaussian

We may use the following formulas to estimate Gaussian parameters (mean and variation) for _i<sup>th</sup>_ feature:

![mu-i](./formulas/mu-i.svg)

![sigma-i](./formulas/sigma-i.svg)

![i](./formulas/i.svg)

![m](./formulas/m.svg) - number of training examples.

![n](./formulas/n.svg) - number of features.

## Density Estimation

So we have a training set:

![Training Set](./formulas/training-set.svg)

![x-in-R](./formulas/x-in-R.svg)

We assume that each feature of the training set is normally distributed:

![x-1](./formulas/x-1.svg)

![x-2](./formulas/x-2.svg)

![x-n](./formulas/x-n.svg)

Then:

![p-x](./formulas/p-x.svg)

![p-x-2](./formulas/p-x-2.svg)

## Anomaly Detection Algorithm

1. Choose features ![x-i](./formulas/x-i.svg) that might be indicative of anomalous examples (![Training Set](./formulas/training-set.svg)).
2. Fit parameters ![params](./formulas/params.svg) using formulas:

![mu-i](./formulas/mu-i.svg)

![sigma-i](./formulas/sigma-i.svg)

3. Given new example _x_, compute _p(x)_:

![p-x-2](./formulas/p-x-2.svg)

Anomaly if ![anomaly](./formulas/anomaly.svg)

![epsilon](./formulas/epsilon.svg) - probability threshold.

## Algorithm Evaluation

The algorithm may be evaluated using _F1_ score.

The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at _1_ (perfect precision and recall) and worst at _0_.

![F1 Score](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)

![f1](./formulas/f1.svg)

Where:

![precision](./formulas/precision.svg)

![recall](./formulas/recall.svg)

_tp_ - number of true positives.

_fp_ - number of false positives.

_fn_ - number of false negatives.

## Files

- [demo.m](./demo.m) - main file that you should run from Octave console in order to see the demo.
- [server_params.mat](./server_params.mat) - training data set.
- [estimate_gaussian.m](./estimate_gaussian.m) - this function estimates the parameters of a Gaussian distribution using the data in X.
- [multivariate_gaussian.m](./multivariate_gaussian.m) - function that computes the probability density function of the multivariate gaussian distribution.
- [select_threshold.m](./select_threshold.m) - function that finds the best threshold (epsilon) to use for selecting outliers.
- [visualize_fit.m](./visualize_fit.m) - Function that visualizes the data set and its estimated distribution.

### Demo visualizations

![Demo visualization](./formulas/demo.png)

## References

- [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
- [Normal Distribution on Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)
- [F1 Score on Wikipedia](https://en.wikipedia.org/wiki/F1_score)
  