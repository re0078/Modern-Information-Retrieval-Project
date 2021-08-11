import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.spatial.distance import cdist


class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k:int, metric:str = 'euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, x, y):
        self.x = x
        self.y = y

    def score(self, x, y):
        P = self.predict(x)
        return np.mean(P == y)
        
    def predict(self, x):
        res = []
        dist = cdist(x, self.x, self.metric)
        dist_index = dist.argsort()[:,:self.k]
        y = np.array(self.y[dist_index])
        y = np.mean(y, axis=1)
        y = (y >= 0.5).astype(int)
        return y