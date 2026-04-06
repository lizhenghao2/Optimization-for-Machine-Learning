import numpy as np
from sigmoid import sigmoid

def compute_loss(A, y, x):
    """
    Logistic regression loss
    """
    y_hat = sigmoid(A @ x)
    eps = 1e-12 # to prevent log(0)
    
    loss = -np.mean(
        y * np.log(y_hat + eps) +
        (1 - y) * np.log(1 - y_hat + eps)
    )
    
    return loss