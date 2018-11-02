# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plot
from LinearRegression import LinearRegression

# Load the data.
data = np.genfromtxt('./data/house-prices.csv', delimiter=',')

# Split the by input and output.
x = data[:, 0:2]
y = data[:, 2:]

# Plot the data.
figure = plot.figure()
ax = figure.add_subplot(111, projection='3d')
ax.scatter(x[:, :1], x[:, 1:2], y, c='r', marker='o')
ax.set_xlabel('Size')
ax.set_ylabel('Rooms')
ax.set_zlabel('Price')
plot.show()

# Init linear regression.
linear_regression = LinearRegression(x, y, alpha=0.1, lambda_param=0, num_iterations=50)

# Train linear regression.
theta, mu, sigma, x_normalized, j_history = linear_regression.train()

print('- Initial cost: {0}\n'.format(j_history[0]))
print('- Optimized cost: {0}\n'.format(j_history[-1:]))

print('- Theta (with normalization):\n')
print('-- {0}\n'.format(theta))
print('\n')

# Data for plotting
fig, ax = plot.subplots()
ax.plot(range(50), j_history)
plot.show()
