"""
Homemade Gradient Boosting Classifier.

This module provides a from-scratch implementation of Gradient Boosting
for binary classification using log-loss (binary cross-entropy) as the
objective function.

Reference:
    Friedman, J. H. (2001). Greedy function approximation: a gradient
    boosting machine. Annals of Statistics, 29(5), 1189-1232.
"""

import numpy as np


class DecisionTreeRegressor:
    """A simple decision tree for regression (used as a weak learner).

    Each internal node splits on a single feature threshold; leaves store
    the mean target value of samples that fall into them.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    """

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # Internal tree structure stored as a dict after fitting.
        self.tree = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features, targets):
        """Build the regression tree from training data.

        Parameters
        ----------
        features : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        targets : ndarray of shape (n_samples,)
            Continuous target values (pseudo-residuals in boosting context).
        """
        self.tree = self._build_tree(features, targets, depth=0)

    def predict(self, features):
        """Predict target values for each sample in *features*.

        Parameters
        ----------
        features : ndarray of shape (n_samples, n_features)

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
        """
        return np.array([self._traverse(row, self.tree) for row in features])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tree(self, features, targets, depth):
        """Recursively build the tree and return a node dict."""
        n_samples = len(targets)

        # --- Stopping criteria ---
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {"leaf": True, "value": float(np.mean(targets))}

        best = self._best_split(features, targets)
        if best is None:
            return {"leaf": True, "value": float(np.mean(targets))}

        feature_index, threshold = best
        left_mask = features[:, feature_index] <= threshold

        # If the split sends all samples to one side, make a leaf.
        if left_mask.all() or (~left_mask).all():
            return {"leaf": True, "value": float(np.mean(targets))}

        return {
            "leaf": False,
            "feature_index": feature_index,
            "threshold": threshold,
            "left": self._build_tree(
                features[left_mask], targets[left_mask], depth + 1
            ),
            "right": self._build_tree(
                features[~left_mask], targets[~left_mask], depth + 1
            ),
        }

    def _best_split(self, features, targets):  # pylint: disable=too-many-locals
        """Find the (feature, threshold) pair that minimises MSE after split.

        Returns
        -------
        (feature_index, threshold) or None if no valid split exists.
        """
        _, n_features = features.shape
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            column = features[:, feature_index]
            # Candidate thresholds: midpoints between consecutive sorted values.
            sorted_values = np.unique(column)
            if len(sorted_values) < 2:
                continue
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                mse = self._weighted_mse(targets, left_mask, right_mask)
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_index
                    best_threshold = threshold

        if best_feature is None:
            return None
        return best_feature, best_threshold

    @staticmethod
    def _weighted_mse(targets, left_mask, right_mask):
        """Compute the size-weighted MSE of a proposed split."""
        n_total = len(targets)
        left_targets = targets[left_mask]
        right_targets = targets[right_mask]

        mse_left = np.var(left_targets) * len(left_targets)
        mse_right = np.var(right_targets) * len(right_targets)
        return (mse_left + mse_right) / n_total

    def _traverse(self, sample, node):
        """Walk the tree for a single sample and return the leaf value."""
        if node["leaf"]:
            return node["value"]
        if sample[node["feature_index"]] <= node["threshold"]:
            return self._traverse(sample, node["left"])
        return self._traverse(sample, node["right"])


# ---------------------------------------------------------------------------


class GradientBoostingClassifier:
    """Gradient Boosting for binary classification (log-loss objective).

    The model is an additive ensemble of regression trees trained
    sequentially on the negative gradient of the log-loss:

        F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)

    where h_m is a decision tree fitted to the pseudo-residuals

        r_i = y_i - p_i,  with p_i = sigmoid(F_{m-1}(x_i)).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees to add).
    learning_rate : float, default=0.1
        Shrinkage factor applied to each tree's contribution.
    max_depth : int, default=3
        Maximum depth of each weak-learner tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node in each tree.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The fitted weak learners, one per boosting round.
    initial_prediction_ : float
        The log-odds of the positive class used to initialise F_0.
    train_loss_ : list of float
        Log-loss recorded after each boosting round (training set).
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.estimators_ = []
        self.initial_prediction_ = 0.0
        self.train_loss_ = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features, labels):
        """Train the gradient boosting ensemble.

        Parameters
        ----------
        features : ndarray of shape (n_samples, n_features)
        labels   : ndarray of shape (n_samples,)  — binary (0 / 1)
        """
        labels = np.asarray(labels, dtype=float)
        n_samples = len(labels)

        # F_0: initialise with the log-odds of the positive class.
        positive_rate = np.clip(labels.mean(), 1e-15, 1 - 1e-15)
        self.initial_prediction_ = np.log(positive_rate / (1.0 - positive_rate))

        # Running raw score for every training sample.
        raw_scores = np.full(n_samples, self.initial_prediction_)

        self.estimators_ = []
        self.train_loss_ = []

        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(raw_scores)

            # Pseudo-residuals = negative gradient of log-loss w.r.t. raw score.
            pseudo_residuals = labels - probabilities

            # Fit a tree to the pseudo-residuals.
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(features, pseudo_residuals)
            self.estimators_.append(tree)

            # Update raw scores (additive model update).
            raw_scores += self.learning_rate * tree.predict(features)

            # Record training log-loss.
            probabilities = self._sigmoid(raw_scores)
            loss = self._log_loss(labels, probabilities)
            self.train_loss_.append(loss)

        return self

    def predict_proba(self, features):
        """Return class probabilities P(y=0 | x) and P(y=1 | x).

        Parameters
        ----------
        features : ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        raw_scores = self._raw_scores(features)
        prob_positive = self._sigmoid(raw_scores)
        return np.column_stack([1.0 - prob_positive, prob_positive])

    def predict(self, features):
        """Predict binary class labels (0 or 1).

        Parameters
        ----------
        features : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _raw_scores(self, features):
        """Compute the raw additive score F_M(x) for each sample."""
        n_samples = features.shape[0]
        scores = np.full(n_samples, self.initial_prediction_)
        for tree in self.estimators_:
            scores += self.learning_rate * tree.predict(features)
        return scores

    @staticmethod
    def _sigmoid(values):
        """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
        return np.where(
            values >= 0,
            1.0 / (1.0 + np.exp(-values)),
            np.exp(values) / (1.0 + np.exp(values)),
        )

    @staticmethod
    def _log_loss(labels, probabilities):
        """Binary cross-entropy loss (log-loss).

        Parameters
        ----------
        labels        : ndarray of shape (n_samples,)  — 0 / 1
        probabilities : ndarray of shape (n_samples,)  — in (0, 1)

        Returns
        -------
        loss : float
        """
        clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
        return -np.mean(
            labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)
        )
