import numpy as np


def array32(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


def zeros32(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.zeros(*args, **kwargs)


def ones32(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.ones(*args, **kwargs)
