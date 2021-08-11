import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import numpy as np


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, c: int = 1.0, kernel: str = 'rbf'):
        self.c = c
        self.kernel = kernel
       
    def fit(self, x, y):
        clf = SVC(C=self.c, kernel=self.kernel)
        clf.fit(x, y)
        self.clf = clf
        return self
    
    def score(self, x, y):
        p = self.predict(x)
        return np.mean(p == y)

    def predict(self, x, y=None):
        return self.clf.predict(x)