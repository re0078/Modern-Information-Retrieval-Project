import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=50, hidden_layer_sizes=100, activation='relu', solver='adam'):
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver

    def fit(self, x, y):
        clf = MLPClassifier(
            max_iter=self.max_iter, hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver = self.solver
        )
        clf.fit(x, y)
        self.clf = clf
        return self

    def predict(self, x):
        return self.clf.predict(x)
