from k_means import KMeans
import pandas as pd

# Load data.
data = pd.read_csv('homemade-machine-learning\data\iris.csv')

# Get total number of Iris examples.
num_examples = data.shape[0]

# Get features.
x_train = data[['petal_length', 'petal_width']].values.reshape((num_examples, 2))

# Set K-Means parameters.

# Clusters act as the cassifications for our data.
num_clusters = 3

# Max number of iterations, acts as a stopping condition.
max_iterations = 50

# Load KMeans.
k_means = KMeans(x_train, num_clusters)

# Train / fit K-Means.
(centroids, closest_centroids_ids) = k_means.train(max_iterations)

print(centroids)

# There is no prediction module?
