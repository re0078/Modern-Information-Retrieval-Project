import typing as th
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.cluster import AgglomerativeClustering
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


class Hierarchical(ClusterMixin, BaseEstimator):
    def __init__(self, cluster_count: int=8):
        self.cluster_count = cluster_count
        pass

    def fit_predict(self, x, **kwargs):
        self.model = AgglomerativeClustering(n_clusters=self.cluster_count)
        return self.model.fit_predict(x)
    
    def score(self, x, y):
        return purity_score(y, self.fit_predict(x))