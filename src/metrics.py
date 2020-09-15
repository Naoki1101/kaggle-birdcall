import numpy as np
from sklearn import metrics


def f1_score(y_true, y_pred, average='micro'):
    return metrics.f1_score(y_true, y_pred, average='micro')
