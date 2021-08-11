import typing as th  # Literals are available for python>=3.8
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            # kind,  #: th.Literal['gaussian', 'bernoulli', ],
            # add required hyper-parameters (if any)
    ):
        # todo: initialize parameters
        pass

    def fit(self, x, y, alpha=0.005):
        print(x)
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            cov , mean = self.make_cov_mean(current_x, current_x.shape[1], alpha=0.005)
            self.gaussians[c] = {
                'mean': mean,
                'cov': cov
            }  
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
        return self

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, cov = g['mean'] , g['cov']
            det_value = np.linalg.slogdet(cov)[1]
            cov_inv = np.linalg.inv(cov)
            for i in range(len(X)):
                P[i,c] = np.log(self.priors[c]) - (1/2)*(X[i] - mean).T @ cov_inv @ (X[i] - mean)
        return np.argmax(P, axis=1)

    def make_cov_mean(self, X, lent, alpha=0.005):
        X = np.array(X) 
        covariance = np.cov(X.T) + alpha * np.identity(X.shape[1]) 
        mean = np.mean(X.T, axis=1)
        return covariance, mean