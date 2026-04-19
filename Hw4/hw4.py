import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gradient_svm import svm_gradient



# load csv data to pandas dataframe
data = pd.read_csv('social.csv')
print(data.head())
print(data.info())

# split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.25, random_state=41)

# standardize the features
train_data['Age'] = (train_data['Age'] - train_data['Age'].mean()) / train_data['Age'].std()
train_data['EstimatedSalary'] = (train_data['EstimatedSalary'] - train_data['EstimatedSalary'].mean()) / train_data['EstimatedSalary'].std()
test_data['Age'] = (test_data['Age'] - test_data['Age'].mean()) / test_data['Age'].std()
test_data['EstimatedSalary'] = (test_data['EstimatedSalary'] - test_data['EstimatedSalary'].mean()) / test_data['EstimatedSalary'].std()

# convert the purchased variable to labels -1 and 1
y = 2 * train_data['Purchased'].values - 1
x = train_data[['Age', 'EstimatedSalary']].values
print('Training data shape:', x.shape)
print('Gender is not used as a feature, only Age and EstimatedSalary are used.')

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# gradient descent
threshold = 1e-5
learning_rate = 0.01
max_iters = 1000

# test data (shared across all C values)
y_test = 2 * test_data['Purchased'].values - 1
x_test = test_data[['Age', 'EstimatedSalary']].values

for C in [0.1, 1, 10, 50, 100]:
    # theta = [w;b] where w is the weight vector and b is the bias term
    theta = np.zeros(x.shape[1] + 1)

    for i in range(max_iters):
        grad = svm_gradient(theta, x, y, C)
        theta -= learning_rate * grad
        if np.linalg.norm(grad) < threshold:
            print(f'C={C}: Convergence reached at iteration {i}')
            break

    predictions = x_test.dot(theta[:-1]) + theta[-1]
    predicted_labels = np.where(predictions >= 0, 1, -1)
    accuracy = np.mean(predicted_labels == y_test)
    cm = confusion_matrix(y_test, predicted_labels, labels=[-1, 1])

    print(f'\n=== Results for C={C} ===')
    print('Final parameters:', theta)
    print('Test set accuracy:', accuracy)
    print('Confusion Matrix (rows=true, cols=pred; labels=[-1, 1]):')
    print(cm)
    
    
    
    # plot the training data points
    plt.figure(figsize=(8, 6))
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='g', edgecolors='k', label='y=1 (Purchased)')
    plt.scatter(x[y == -1, 0], x[y == -1, 1], c='r', edgecolors='k', label='y=-1 (Not Purchased)')
    
    # Plot hyperplane
    x_min, x_max = x[:, 0].min() , x[:, 0].max()
    xx = np.linspace(x_min, x_max, 100)
    yy = -(theta[0] * xx + theta[2]) / theta[1]
    plt.plot(xx, yy, 'b-', linewidth=2, label='Decision Boundary')
    
    plt.xlabel('Age ')
    plt.ylabel('Estimated Salary ')
    plt.title(f'SVM Decision Boundary on Training Data (C={C})')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, f'svm_train_C_{C}.png'))
    
    # plot the test data points
    plt.figure(figsize=(8, 6))
    plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='g', edgecolors='k', label='y=1 (Purchased)')
    plt.scatter(x_test[y_test == -1, 0], x_test[y_test == -1, 1], c='r', edgecolors='k', label='y=-1 (Not Purchased)')
    
    # Plot hyperplane
    x_min, x_max = x_test[:, 0].min() , x_test[:, 0].max() 
    xx = np.linspace(x_min, x_max, 100)
    yy = -(theta[0] * xx + theta[2]) / theta[1]
    plt.plot(xx, yy, 'b-', linewidth=2, label='Decision Boundary')
    
    plt.legend()
    plt.xlabel('Age ')
    plt.ylabel('Estimated Salary ')
    plt.title(f'SVM Decision Boundary with Test Data (C={C})')
    plt.savefig(os.path.join(FIGURES_DIR, f'svm_test_C_{C}.png'))