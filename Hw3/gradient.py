# gradient.py
import numpy as np
from sigmoid import sigmoid

def compute_gradient(A, y, x):
    """
    Compute gradient of logistic regression loss.

    Parameters:
        A : np.ndarray, shape (N, d)
        y : np.ndarray, shape (N,)
        x : np.ndarray, shape (d,)

    Returns:
        grad : np.ndarray, shape (d,)
    """
    N = len(y)
    
    y_hat = sigmoid(A @ x)           # (N,)
    grad = (A.T @ (y_hat - y)) / N   # (d,)
    
    return grad