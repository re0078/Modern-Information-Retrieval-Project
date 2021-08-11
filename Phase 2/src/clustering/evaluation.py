import typing as th
from sklearn.metrics import adjusted_rand_score
import numpy as np

def purity(y, y_hat) -> float:
    n = len(y)
    confusion_matrix = np.zeros((n , n))
    for i in range(n):
        true_index = y[i]
        predicted_index = y_hat[i]
        confusion_matrix[true_index, predicted_index] += 1
    purity = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
    return purity


def adjusted_rand_index(y, y_hat) -> float:
    return adjusted_rand_score(y, y_hat)


evaluation_functions = dict(purity=purity, adjusted_rand_index=adjusted_rand_index)


def evaluate(y, y_hat) -> th.Dict[str, float]:
    """
    :param y: ground truth
    :param y_hat: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    return {name: func(y, y_hat) for name, func in evaluation_functions.items()}
