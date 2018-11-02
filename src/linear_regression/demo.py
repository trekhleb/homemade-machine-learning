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

# Plot the data.
FIGURE = plot.figure()
AX = FIGURE.add_subplot(111, projection='3d')
AX.scatter(X[:, :1], X[:, 1:2], Y, c='r', marker='o')
AX.set_xlabel('Size')
AX.set_ylabel('Rooms')
AX.set_zlabel('Price')
plot.show()

# Init linear regression.
LINEAR_REGRESSION = LinearRegression(X, Y)

# Train linear regression.
(
    THETA,
    FEATURES_MEAN,
    FEATURES_DEVIATION,
    TRAINING_SET_NORMALIZED,
    COST_HISTORY
) = LINEAR_REGRESSION.train(alpha=0.1, lambda_param=0, num_iterations=50)

print('- Initial cost: {0}\n'.format(COST_HISTORY[0]))
print('- Optimized cost: {0}\n'.format(COST_HISTORY[-1:]))

print('- Theta (with normalization):\n')
print('-- {0}\n'.format(THETA))
print('\n')

# Data for plotting
FIG, AX = plot.subplots()
AX.plot(range(50), COST_HISTORY)
plot.show()
