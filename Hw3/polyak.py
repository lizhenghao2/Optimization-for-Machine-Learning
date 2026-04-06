# Polyak's momentum method implementation
import numpy as np
from gradient import compute_gradient
from loss import compute_loss
def polyak_momentum(A, y, alpha=0.01, beta=0.9, max_iters=1000, tol=1e-6):
    """
    Perform Polyak's momentum method for logistic regression.

    Parameters:
        A : np.ndarray, shape (N, d)
        y : np.ndarray, shape (N,)
        alpha: step size (learning rate)
        beta: momentum parameter
        max_iters : int, maximum number of iterations
        tol : float, tolerance for stopping (based on gradient norm)
    Returns:
        x : np.ndarray, shape (d,)
        history : list of loss values
    """
    x = np.zeros(A.shape[1])   # initialize parameters
    v = np.zeros_like(x)      # initialize velocity
    x_history = []
    loss_history = []

    for i in range(max_iters):
        grad = compute_gradient(A, y, x)
        grad_norm = np.linalg.norm(grad)
        
        curr_loss = compute_loss(A, y, x)
        loss_history.append(curr_loss)

        if grad_norm < tol:
            print(f"Converged at iteration {i}")
            x_history.append(x.copy())
            break

        x_prev = x.copy()
        x = x - alpha * grad + beta * v
        v = x - x_prev  # update velocity
        x_history.append(x.copy())

    return x, x_history, loss_history