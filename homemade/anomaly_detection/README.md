# Anomaly Detection Using Gaussian Distribution

## Jupyter Demos

▶️ [Demo | Anomaly Detection](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/anomaly_detection/anomaly_detection_gaussian_demo.ipynb) - find anomalies in server operational parameters like `latency` and `threshold`

## Gaussian (Normal) Distribution

The **normal** (or **Gaussian**) **distribution** is a very common continuous probability distribution. Normal distributions are important in statistics and are often used in the natural and social sciences to represent real-valued random variables whose distributions are not known. A random variable with a Gaussian distribution is said to be normally distributed and is called a normal deviate.

Let's say:

![x-in-R](../../images/anomaly_detection/x-in-R.svg)

If _x_ is normally distributed then it may be displayed as follows.

![Gaussian Distribution](https://upload.wikimedia.org/wikipedia/commons/7/74/Normal_Distribution_PDF.svg)

![mu](../../images/anomaly_detection/mu.svg) - mean value,

![sigma-2](../../images/anomaly_detection/sigma-2.svg) - variance.

![x-normal](../../images/anomaly_detection/x-normal.svg) - "~" means that _"x is distributed as ..."_

Then Gaussian distribution (probability that some _x_ may be a part of distribution with certain mean and variance) is given by:

![Gaussian Distribution](../../images/anomaly_detection/p.svg)

## Estimating Parameters for a Gaussian

We may use the following formulas to estimate Gaussian parameters (mean and variation) for _i<sup>th</sup>_ feature:

![mu-i](../../images/anomaly_detection/mu-i.svg)

![sigma-i](../../images/anomaly_detection/sigma-i.svg)

![i](../../images/anomaly_detection/i.svg)

![m](../../images/anomaly_detection/m.svg) - number of training examples.

![n](../../images/anomaly_detection/n.svg) - number of features.

## Density Estimation

So we have a training set:

![Training Set](../../images/anomaly_detection/training-set.svg)

![x-in-R](../../images/anomaly_detection/x-in-R.svg)

We assume that each feature of the training set is normally distributed:

![x-1](../../images/anomaly_detection/x-1.svg)

![x-2](../../images/anomaly_detection/x-2.svg)

![x-n](../../images/anomaly_detection/x-n.svg)

Then:

![p-x](../../images/anomaly_detection/p-x.svg)

![p-x-2](../../images/anomaly_detection/p-x-2.svg)

## Anomaly Detection Algorithm

1. Choose features ![x-i](../../images/anomaly_detection/x-i.svg) that might be indicative of anomalous examples (![Training Set](../../images/anomaly_detection/training-set.svg)).
2. Fit parameters ![params](../../images/anomaly_detection/params.svg) using formulas:

![mu-i](../../images/anomaly_detection/mu-i.svg)

![sigma-i](../../images/anomaly_detection/sigma-i.svg)

3. Given new example _x_, compute _p(x)_:

![p-x-2](../../images/anomaly_detection/p-x-2.svg)

Anomaly if ![anomaly](../../images/anomaly_detection/anomaly.svg)

![epsilon](../../images/anomaly_detection/epsilon.svg) - probability threshold.

## Algorithm Evaluation

The algorithm may be evaluated using _F1_ score.

The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at _1_ (perfect precision and recall) and worst at _0_.

![F1 Score](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)

![f1](../../images/anomaly_detection/f1.svg)

Where:

![precision](../../images/anomaly_detection/precision.svg)

![recall](../../images/anomaly_detection/recall.svg)

_tp_ - number of true positives.

_fp_ - number of false positives.

_fn_ - number of false negatives.

## References

- [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
- [Normal Distribution on Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)
- [F1 Score on Wikipedia](https://en.wikipedia.org/wiki/F1_score)
- [Precision and Recall on Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
  