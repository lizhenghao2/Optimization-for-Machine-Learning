import numpy as np
def svm_gradient(theta, x, y, C):
    """
    Compute the gradient of the SVM loss function with respect to theta.
    
    Parameters:
    theta: numpy array of shape (d+1,) where d is the number of features. The last element is the bias term.
    x: numpy array of shape (n, d) where n is the number of samples and d is the number of features.
    y: numpy array of shape (n,) containing the labels (-1 or 1) for each sample.
    C: float, regularization parameter.
    
    Returns:
    grad: numpy array of shape (d+1,) containing the gradient of the loss function with respect to theta.
    """
    n, d = x.shape
    w = theta[:-1]  
    b = theta[-1]   
    
    
    predictions = x.dot(w) + b # <w, x_n> + b
    decision_values = y * predictions # t_n
    
    # Compute the indicator for misclassified samples
    misclassified = decision_values < 1
        
    # Compute the gradient
    grad_w = w - C * np.sum((misclassified * y)[:, np.newaxis] * x, axis=0)
    grad_b = -C * np.sum(misclassified * y)
    
    grad = np.hstack([grad_w, grad_b])
    
    return grad