"""Linear Regression Demo"""

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
import numpy as np
import matplotlib.pyplot as plot
from linear_regression import LinearRegression

# Load the data.
DATA = np.genfromtxt('./data/house-prices.csv', delimiter=',')

# Split the by input and output.
X = DATA[:, 0:2]
Y = DATA[:, 2:]

# Init linear regression.
LINEAR_REGRESSION = LinearRegression(X, Y)

# Train linear regression.
NUM_ITERATIONS = 30
LAMBDA_PARAM = 0
ALPHA = 0.1

(
    THETA,
    FEATURES_MEAN,
    FEATURES_DEVIATION,
    TRAINING_SET_NORMALIZED,
    COST_HISTORY
) = LINEAR_REGRESSION.train(ALPHA, LAMBDA_PARAM, NUM_ITERATIONS)

print(np.min(X[:, 0]))
print(np.min(X[:, 1]))

print('Initial cost: {0}\n'.format(COST_HISTORY[0]))
print('Optimized cost: {0}\n'.format(COST_HISTORY[-1:]))

print('Theta:\n')
print('- {0}\n'.format(THETA))

# Plot the data.
FIGURE = plot.figure(1, figsize=(8, 6))

# Plot the training set.
AX1 = FIGURE.add_subplot(221, projection='3d', title='Training Set')
AX1.scatter(X[:, :1], X[:, 1:2], Y, c='r', marker='o')
AX1.set_xlabel('Size')
AX1.set_ylabel('Rooms')
AX1.set_zlabel('Price')

# Plot normalized training set.
AX2 = FIGURE.add_subplot(222, projection='3d', title='Normalized Training Set')
AX2.scatter(TRAINING_SET_NORMALIZED[:, 1:2], TRAINING_SET_NORMALIZED[:, 2:3], Y, c='r', marker='o')
AX2.set_xlabel('Normalized Size')
AX2.set_ylabel('Normalized Rooms')
AX2.set_zlabel('Price')

# Plot gradient descent progress.
AX3 = FIGURE.add_subplot(223, title='Gradient Descent')
AX3.plot(range(NUM_ITERATIONS), COST_HISTORY)
AX3.set_xlabel('Iterations')
AX3.set_ylabel('Cost')
AX3.grid(False)

plot.suptitle('Linear Regression')
plot.show()
