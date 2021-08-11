import typing as th
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def accuracy(y, y_hat) -> float:
    return accuracy_score(y, y_hat)


def f1(y, y_hat, alpha: float = 0.5, beta: float = 1.):
    return f1_score(y, y_hat)


def precision(y, y_hat) -> float:
    return precision_score(y, y_hat)


def recall(y, y_hat) -> float:
    return recall_score(y, y_hat)


def f1_negative(y, y_hat, alpha: float = 0.5, beta: float = 1.):
    y = 1 - y
    y_hat = 1 - y_hat
    return f1_score(y, y_hat)


def precision_negative(y, y_hat) -> float:
    y = 1 - y
    y_hat = 1 - y_hat
    return precision_score(y, y_hat)


def recall_negative(y, y_hat) -> float:
    y = 1 - y
    y_hat = 1 - y_hat
    return recall_score(y, y_hat)


evaluation_functions = dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall,f1_negative=f1_negative, precision_negative=precision_negative, recall_negative=recall_negative)


def evaluate(y, y_hat) -> th.Dict[str, float]:
    """
    :param y: ground truth
    :param y_hat: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    return {name: func(y, y_hat) for name, func in evaluation_functions.items()}
