import typing as th  # Literals are available for python>=3.8
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class NaiveBayes(BaseEstimator, ClassifierMixin):   
    def __init__(self, kind:str = 'gaussian', alpha:float = 0.005):
        self.kind = kind
        self.alpha = alpha
    
    def make_cov_mean(self, X, lent):
        X = np.array(X) 
        covariance = np.cov(X.T) + self.alpha * np.identity(X.shape[1]) 
        mean = np.mean(X.T, axis=1)
        return covariance, mean

    def fit(self, X, Y):
        self.gaussians = dict()
        self.priors = dict()
        self.likelihood = dict()
        labels = set(Y)
        if self.kind == 'gaussian':
            for c in labels:
                current_x = X[Y == c]
                cov , mean = self.make_cov_mean(current_x, current_x.shape[1])
                self.gaussians[c] = {
                    'mean': mean,
                    'cov': cov
                }  
                self.priors[c] = float(len(Y[Y == c])) / len(Y)
        elif self.kind == 'bernoulli':
            X_mean = X.mean(axis=0)
            X_new = np.zeros(X.shape)
            for i in range (X.shape[0]):
                X_new[i, :] = (X[i, :] > X_mean).astype(int)
            for c in labels:
                current_x = X_new[Y == c]
                self.likelihood[c] = current_x.mean(axis=0) 
                self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        if self.kind == 'gaussian':
            K = len(self.gaussians)
            P = np.zeros((N, K))
            for c, g in self.gaussians.items():
                mean, cov = g['mean'] , g['cov']
                det_value = np.linalg.slogdet(cov)[1]
                cov_inv = np.linalg.inv(cov)
                for i in range(len(X)):
                    P[i,c] = np.log(self.priors[c]) - (1/2)*(X[i] - mean).T @ cov_inv @ (X[i] - mean)
        elif self.kind == 'bernoulli':
            K = len(self.likelihood)
            P = np.zeros((N, K))
            X_mean = X.mean(axis=0)
            X_new = np.zeros(X.shape)
            for i in range (X.shape[0]):
                X_new[i, :] = (X[i, :] > X_mean).astype(int)
            for c, l in self.likelihood.items():
                for i in range(len(X_new)):
                    P[i, c] = np.log(self.priors[c]) + sum(np.log(np.multiply(l, X_new[i]) + np.multiply(1-l, 1-X_new[i])))
        return np.argmax(P, axis=1)