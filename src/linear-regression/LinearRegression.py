import numpy as np


class LinearRegression:
    def __init__(self, x, y, alpha, lambda_param, num_iterations):
        m, n = x.shape

        x_normalized, mu, sigma = self.normalize_features(x)
        x_normalized = np.hstack((np.ones((m, 1)), x_normalized))

        self.x = x_normalized
        self.y = y
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.m = m
        self.n = n
        self.x_normalized = x_normalized
        self.mu = mu
        self.sigma = sigma

    def train(self):
        initial_theta = np.zeros((self.n + 1, 1))
        theta, j_history = self.gradient_descent(initial_theta)
        return theta, self.mu, self.sigma, self.x, j_history

    def gradient_descent(self, theta):
        j_history = np.zeros((self.num_iterations, 1))
        for iteration in range(self.num_iterations):
            theta = self.gradient_step(theta)
            j_history[iteration] = self.cost_function(theta)
        return theta, j_history

    def gradient_step(self, theta):
        predictions = self.hypothesis(theta)
        difference = predictions - self.y
        regularization_param = 1 - self.alpha * self.lambda_param / self.m
        theta = theta * regularization_param - self.alpha * (1 / self.m) * (difference.T @ self.x).T
        theta[1] = theta[1] - self.alpha * (1 / self.m) * (self.x[:, 0].T @ difference).T
        return theta

    def cost_function(self, theta):
        differences = self.hypothesis(theta) - self.y
        theta_cut = theta[1:, 0]
        regularization_param = self.lambda_param * (theta_cut.T @ theta_cut)
        cost = (1 / 2 * self.m) * (differences.T @ differences + regularization_param)
        return cost

    def hypothesis(self, theta):
        predictions = self.x @ theta
        return predictions

    def normalize_features(self, x):
        """Normalize input features

        Normalizes the features in x. Returns a normalized version of x where the mean value of
        each feature is 0 and the standard deviation is 1.
        """
        x_normalized = x
        mu = np.mean(x)
        sigma = np.std(x)
        x_normalized -= mu
        x_normalized /= sigma
        return x_normalized, mu, sigma
