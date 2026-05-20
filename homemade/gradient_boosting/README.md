# Gradient Boosting

## References

- 📗 [Greedy Function Approximation: A Gradient Boosting Machine – Friedman (2001)](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
- 🔗 [Wikipedia – Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)

## Overview

**Gradient Boosting** is a supervised ensemble technique that builds a strong predictive model by adding weak learners (usually shallow decision trees) one at a time, where each new tree corrects the errors of the existing ensemble.

Unlike Random Forests, which train trees in parallel, Gradient Boosting trains trees **sequentially**, guided by the **gradient of the loss function**.

## Demo

➡️ [Open in Jupyter Notebook](../../notebooks/gradient_boosting_demo.ipynb)

Dataset: `server-operational-params.csv` — predict server anomalies (binary classification) from latency and throughput features.

## The Model

The ensemble prediction is an additive model:

```
F_M(x) = F_0 + Σ_{m=1}^{M} η · h_m(x)
```

where:
- `F_0` is the initial prediction (log-odds of the positive class),
- `h_m` is the *m*-th weak learner (regression tree),
- `η` (learning rate) is a shrinkage factor that controls each tree's contribution,
- `M` is the total number of boosting rounds (`n_estimators`).

## Loss Function – Log-Loss (Binary Cross-Entropy)

For binary classification we minimise log-loss over the training set:

```
L(y, p) = -1/n · Σ_{i=1}^{n} [ y_i log(p_i) + (1 − y_i) log(1 − p_i) ]
```

where `p_i = sigmoid(F(x_i))`.

## Pseudo-Residuals (Negative Gradient)

At each boosting round *m*, we compute the **pseudo-residuals** — the negative gradient of *L* with respect to the current raw score `F_{m-1}(x_i)`:

```
r_i = − ∂L/∂F(x_i) = y_i − p_i
```

For log-loss the pseudo-residual is simply the difference between the true label and the current predicted probability. This is what each new tree is trained to approximate.

## Algorithm

```
1. Initialise:  F_0(x) = log( p̄ / (1 − p̄) )   # log-odds of mean label

2. For m = 1, 2, …, M:
   a. Compute pseudo-residuals:
          r_i = y_i − sigmoid( F_{m-1}(x_i) )   for i = 1 … n

   b. Fit a regression tree h_m to {(x_i, r_i)}.

   c. Update the ensemble:
          F_m(x) = F_{m-1}(x) + η · h_m(x)

3. Predict:  ŷ = 1  if  sigmoid( F_M(x) ) ≥ 0.5  else  0
```

## Key Hyperparameters

| Parameter | Description |
|---|---|
| `n_estimators` | Number of boosting rounds (trees). More rounds → lower training loss, risk of overfitting. |
| `learning_rate` (η) | Shrinks each tree's contribution. Smaller values require more trees but generalise better. |
| `max_depth` | Depth of each tree. Shallow trees (2–5) act as weak learners; deeper trees can overfit. |
| `min_samples_split` | Minimum samples needed to split a node; regularises tree growth. |

## Relationship Between Learning Rate and n_estimators

A lower `learning_rate` requires a larger `n_estimators` to achieve the same training loss — but often yields better generalisation (the classic **shrinkage-trees trade-off**).

## Strengths and Limitations

| Strengths | Limitations |
|---|---|
| Handles non-linear boundaries naturally | Sequential training is slow to parallelise |
| Robust to outliers (with robust loss functions) | Many hyperparameters to tune |
| Often achieves state-of-the-art on tabular data | Can overfit on small datasets with high `n_estimators` |
| Built-in feature importance via tree splits | Interpretability decreases as ensemble grows |
