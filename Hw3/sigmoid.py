import numpy as np

def sigmoid(z):
    """
    Sigmoid function applied element-wise.

    Parameters:
        z : np.ndarray or float

    Returns:
        np.ndarray or float
    """
    return 1 / (1 + np.exp(-z))