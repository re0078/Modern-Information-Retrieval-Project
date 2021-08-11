import typing as th
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
import numpy as np

def purity_score(y, y_hat):
    n = len(y)
    confusion_matrix = np.zeros((n , n))
    for i in range(n):
        true_index = y[i]
        predicted_index = y_hat[i]
        confusion_matrix[true_index, predicted_index] += 1
    purity = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
    return purity


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(self, cluster_count: int=8, max_iteration: int=300):
        self.cluster_count = cluster_count
        self.max_iteration = max_iteration
        self.centroids = {}

    def fit(self, x):
        for i in range(self.cluster_count):
            random_point = np.random.randint(0, len(x))
            self.centroids[i] = x[random_point]
        for i in range(self.max_iteration):
            k_data = {}
            for j in range(self.cluster_count):
                k_data[j] = []
            for data in x:
                data_class = self.cluster_predict(data)
                k_data[data_class].append(data)
            pre_centroids = dict(self.centroids)
            for j in range(self.cluster_count):
                self.centroids[j] = np.average(k_data[j], axis=0)
        
        return self

    def cluster_predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[j]) for j in range(self.cluster_count)]
        data_class = distances.index(min(distances))
        return data_class
    
    def predict(self, x):
        result = []
        for data in x:
            data_class = self.cluster_predict(data)
            result.append(data_class)
        return result
    
    def score(self, x, y):
        return purity_score(y, self.predict(x))