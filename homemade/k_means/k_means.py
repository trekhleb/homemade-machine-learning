import numpy as np


class KMeans:
    """K-Means Class"""

    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        centroids = KMeans.init_centroids(self.data, self.num_clusters)

        # Run K-Means.
        for iteration_index in range(max_iterations):
            # Find the closest centroids for training examples.
            closest_centroids_ids = KMeans.find_closest_centroids(self.data, centroids)

    @staticmethod
    def init_centroids(data, num_clusters):
        """Initializes num_clusters centroids that are to be used in K-Means on the dataset data"""

        # Get number of training examples.
        num_examples = data.shape[0]

        # Randomly reorder indices of training examples.
        random_ids = np.random.permutation(num_examples)

        # Take the first K examples as centroids.
        centroids = data[random_ids[:num_clusters + 1], :]

        # Return generated centroids.
        return centroids

    @staticmethod
    def find_closest_centroids(data, centroids):
        # Get number of training examples.
        num_examples = data.shape[0]

        # Get number of centroids.
        num_centroids = centroids.shape[0]
        print(num_centroids)

        # We need to return the following variables correctly.
        closest_centroids_ids = np.zeros((num_examples, 1))

        # Go over every example, find its closest centroid, and store
        # the index inside closest_centroids_ids at the appropriate location.
        # Concretely, closest_centroids_ids(i) should contain the index of the centroid
        # closest to example i. Hence, it should be a value in the range 1...num_centroids.
        for example_index in range(num_examples):
            distances = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):
                distance_difference = data[example_index, :] - centroids[centroid_index, :]
                distances[centroid_index] = np.sum(distance_difference ** 2)
            print(distances)
            print(np.argmin(distances))
            break