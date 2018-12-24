"""KMeans Module"""

import numpy as np


class KMeans:
    """K-Means Class"""

    def __init__(self, data, num_clusters):
        """K-Means class constructor.

        :param data: training dataset.
        :param num_clusters: number of cluster into which we want to break the dataset.
        """
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        """Function performs data clustering using K-Means algorithm

        :param max_iterations: maximum number of training iterations.
        """

        # Generate random centroids based on training set.
        centroids = KMeans.centroids_init(self.data, self.num_clusters)

        # Init default array of closest centroid IDs.
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))

        # Run K-Means.
        for _ in range(max_iterations):
            # Find the closest centroids for training examples.
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)

            # Compute means based on the closest centroids found in the previous part.
            centroids = KMeans.centroids_compute(
                self.data,
                closest_centroids_ids,
                self.num_clusters
            )

        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_init(data, num_clusters):
        """Initializes num_clusters centroids that are to be used in K-Means on the dataset X

        :param data: training dataset.
        :param num_clusters: number of cluster into which we want to break the dataset.
        """

        # Get number of training examples.
        num_examples = data.shape[0]

        # Randomly reorder indices of training examples.
        random_ids = np.random.permutation(num_examples)

        # Take the first K examples as centroids.
        centroids = data[random_ids[:num_clusters], :]

        # Return generated centroids.
        return centroids

    @staticmethod
    def centroids_find_closest(data, centroids):
        """Computes the centroid memberships for every example.

        Returns the closest centroids in closest_centroids_ids for a dataset X where each row is
        a single example. closest_centroids_ids = m x 1 vector of centroid assignments (i.e. each
        entry in range [1..K]).

        :param data: training dataset.
        :param centroids: list of centroid points.
        """

        # Get number of training examples.
        num_examples = data.shape[0]

        # Get number of centroids.
        num_centroids = centroids.shape[0]

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
            closest_centroids_ids[example_index] = np.argmin(distances)

        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clusters):
        """Compute new centroids.

        Returns the new centroids by computing the means of the data points assigned to
        each centroid.

        :param data: training dataset.
        :param closest_centroids_ids: list of closest centroid ids per each training example.
        :param num_clusters: number of clusters.
        """

        # Get number of features.
        num_features = data.shape[1]

        # We need to return the following variables correctly.
        centroids = np.zeros((num_clusters, num_features))

        # Go over every centroid and compute mean of all points that
        # belong to it. Concretely, the row vector centroids(i, :)
        # should contain the mean of the data points assigned to
        # centroid i.
        for centroid_id in range(num_clusters):
            closest_ids = closest_centroids_ids == centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)

        return centroids
