# Logistic Regression for Predicting Customer Behavior

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from grad_descent import gradient_descent
from sigmoid import sigmoid
from polyak import polyak_momentum

# load data
simmons_data = pd.read_csv('simmons2.csv')

print('Shape of the data:', simmons_data.shape)
print(simmons_data.head())
print('Data types:\n', simmons_data.dtypes)

# N is the number of samples
N = simmons_data.shape[0]

# Construct matrix A
A = np.column_stack((
    np.ones(N),
    simmons_data['Spends'],
    simmons_data['HasCard']
)).astype(float)
# print('First 5 rows of matrix A:\n', A[:5])

y = simmons_data['UsesCoupon'].values.astype(float)
print('Shape of y:', y.shape)

# Run gradient descent
x_opt, x_history, loss_history = gradient_descent(A, y, alpha=0.01, max_iters=1000)
print('Learned parameters:', x_opt)

# Plot loss history
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History during Gradient Descent')
plt.legend()
plt.grid()
plt.savefig('loss_history.png')

# plot a 3D figure to show the x history
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
x1_history = [x[0] for x in x_history]
x2_history = [x[1] for x in x_history]
x3_history = [x[2] for x in x_history]
iters_gd = np.arange(len(x1_history))
ax.plot(x1_history, x2_history, x3_history, color='lightgray', linewidth=1, alpha=0.8)
scatter_gd = ax.scatter(x1_history, x2_history, x3_history, c=iters_gd, cmap='viridis', s=10)
ax.set_xlabel('x0 (Intercept)')
ax.set_ylabel('x1 (Spends Coefficient)')
ax.set_zlabel('x2 (HasCard Coefficient)')
ax.set_title('Parameter Trajectory during Gradient Descent')
fig.colorbar(scatter_gd, ax=ax, pad=0.12, label='Iteration')
plt.legend(['Trajectory'])
plt.savefig('parameter_trajectory.png')

# predict 
spend_levels = np.arange(1,8)
has_card = np.array([0, 1])

predictions = np.zeros((len(has_card), len(spend_levels)))
for j, card in enumerate(has_card):
    for i, spend in enumerate(spend_levels):
        features = np.array([1, spend, card], dtype=float)
        predictions[j, i] = sigmoid(features @ x_opt)
        
print('Predicted probabilities of using coupon:\n', np.round(predictions, 2))
# save predictions to csv
pred_df = pd.DataFrame(predictions, columns=[f'Spends={s}' for s in spend_levels], index=[f'HasCard={c}' for c in has_card])
pred_df.to_csv('coupon_predictions.csv')

# Implement Polyak's momentum method
x_polyak, x_polyak_history, loss_polyak_history = polyak_momentum(A, y, alpha=0.01, beta=0.9, max_iters=1000)
print('Learned parameters with Polyak momentum:', x_polyak)

# Plot loss history
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Gradient Descent Loss')
plt.plot(loss_polyak_history, label='Polyak Momentum Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History during Polyak Momentum')
plt.legend()
plt.grid()
plt.savefig('loss_history_polyak.png')

# plot a 3D figure to show the x history
# mark the sequence of points with different colors to show the trajectory more clearly
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
x1_history = [x[0] for x in x_polyak_history]
x2_history = [x[1] for x in x_polyak_history]
x3_history = [x[2] for x in x_polyak_history]
iters = np.arange(len(x1_history))
ax.plot(x1_history, x2_history, x3_history, color='lightgray', linewidth=1, alpha=0.8)
scatter = ax.scatter(x1_history, x2_history, x3_history, c=iters, cmap='viridis', s=10)
ax.set_xlabel('x0 (Intercept)')
ax.set_ylabel('x1 (Spends Coefficient)')
ax.set_zlabel('x2 (HasCard Coefficient)')
ax.set_title('Parameter Trajectory during Polyak Momentum')
fig.colorbar(scatter, ax=ax, pad=0.12, label='Iteration')
plt.legend(['Trajectory'])
plt.savefig('parameter_trajectory_polyak.png')