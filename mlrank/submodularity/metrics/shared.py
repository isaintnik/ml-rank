import numpy as np
from sklearn.metrics import log_loss


def silent_log_loss(a, b):
    try:
        return log_loss(a, b)
    except:
        # in case that there is only one label in target
        # TODO: have no idea how it should work

        a = np.ones_like(a)
        b = np.ones_like(b)

        a = np.append(a, np.zeros((a.shape[0], 1)), axis=1)
        b = np.append(b, np.zeros((a.shape[0], 1)), axis=1)

        return log_loss(a, b)
