import typing as th
from abc import ABCMeta
from sklearn.base import DensityMixin, BaseEstimator
from sklearn.mixture import GaussianMixture
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


class GMM(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self, cluster_count: int=8, max_iteration: int=200):
        self.cluster_count = cluster_count
        self.max_iteration = max_iteration
        
    def fit(self, x):
        self.model = GaussianMixture(n_components=self.cluster_count, max_iter=self.max_iteration)
        self.model.fit(x)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return purity_score(y, self.predict(x))