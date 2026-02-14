import pandas as pd
import numpy as np
from scipy.linalg import qr, solve_triangular
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('house.csv', index_col=0)

# Print the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nShape of the dataset:")
print(df.shape)

# Extract features and target variable
features = df[['bd', 'sqft']].values
# Construct matrix A
A = np.column_stack((np.ones(features.shape[0]), features))

# Keep first 1000000 rows for faster computation
A = A[:1000000, :]
df = df.iloc[:1000000]

# Normalize features to prevent overflow (keep bias column unchanged)
A_normalized = A.copy().astype(float)
# Normalize only the feature columns (columns 1 and 2), keep bias column
A_mean = A[:, 1:].mean(axis=0)
A_std = A[:, 1:].std(axis=0)
A_std[A_std == 0] = 1  # Avoid division by zero
A_normalized[:, 1:] = (A[:, 1:] - A_mean) / A_std

y = df['price'].values
y_mean = y.mean()
y_std = y.std()
y_normalized = (y - y_mean) / y_std

# Gradient descent parameters
step_sizes = [0.5]
max_iters = 500
# Initialize x
x = np.zeros(A_normalized.shape[1])

# Gradient threshold
threshold = 1e-2

# Keep track of x and f(x) history for plotting
x_history_1 = [x.copy()]


print("\nStarting gradient descent with alpha =", step_sizes[0])
for i in range(max_iters):
    # Compute gradient using normalized data
    grad = - 1 / A_normalized.shape[0] * A_normalized.T @ (y_normalized - A_normalized @ x)
    # print(f"Iteration {i+1}: gradient norm = {np.linalg.norm(grad):.6e}")
    # Check for convergence
    grad_norm = np.linalg.norm(grad)
    if np.isnan(grad_norm) or np.isinf(grad_norm):
        print(f"Warning: Non-finite gradient at iteration {i+1}")
        break
    if grad_norm < threshold:
        print(f"Converged after {i+1} iterations.")
        break
    
    # Update x
    x -= step_sizes[0] * grad
    x_history_1.append(x.copy())
    # Print progress every 100 iterations
    if (i + 1) % 100 == 0:
        print(f"Iteration {i+1}: gradient norm = {grad_norm:.6e}")

print("\nNormalized Learned parameters:")
print(x)

# Recover original scale parameters
# A_mean and A_std now only contain feature columns (indices 0 and 1)
# x is [x0 (bias), x1 (bd coeff), x2 (sqft coeff)]
theta_0 = x[0] * y_std + y_mean - (x[1] * A_mean[0] / A_std[0]) * y_std - (x[2] * A_mean[1] / A_std[1]) * y_std
theta_1 = x[1] * y_std / A_std[0]
theta_2 = x[2] * y_std / A_std[1]
theta = np.array([theta_0, theta_1, theta_2])
print("\nRecovered parameters:")
print(theta)



# Use least squares to find the solution with normalized data (more robust)
x_qr = np.linalg.lstsq(A_normalized, y_normalized, rcond=None)[0]
print("\nParameters from actual solution (normalized data):")
print(x_qr)

# Compare the two sets of parameters
print("\nDifference between gradient descent and QR solution (normalized):")
print(x - x_qr)

x_1 = x_qr[0]

# Recovery of original scale parameters from QR solution
theta_qr_0 = x_qr[0] * y_std + y_mean - (x_qr[1] * A_mean[0] / A_std[0]) * y_std - (x_qr[2] * A_mean[1] / A_std[1]) * y_std
theta_qr_1 = x_qr[1] * y_std / A_std[0]
theta_qr_2 = x_qr[2] * y_std / A_std[1]
theta_qr = np.array([theta_qr_0, theta_qr_1, theta_qr_2])
print("\nRecovered parameters from QR solution:")

# save normalized dataset
normalized_df = pd.DataFrame(A_normalized, columns=['bias', 'bd_normalized', 'sqft_normalized'])
normalized_df['price_normalized'] = y_normalized
normalized_df.to_csv('normalized_house.csv', index=False)

# save parameters to a csv
params_df = pd.DataFrame({
    'parameter': ['x_norm', 'x_gd', 'x_qr','x_qr_recovered'],
    'x1': [x[0], theta[0], x_qr[0], theta_qr[0]],
    'x2': [x[1], theta[1], x_qr[1], theta_qr[1]],
    'x3': [x[2], theta[2], x_qr[2], theta_qr[2]]
})
params_df.to_csv('parameters.csv')

# Plot level sets of the objective function (using normalized data)
# Objective function: f(x) = 1/(2n) * ||y_normalized - A_normalized @ x||^2
x1_fixed = x[0]

# Create a grid for x2 and x3 (normalized parameters)
x2_range = np.linspace(x_qr[1] - 0.5, x_qr[1] + 0.5, 10)
x3_range = np.linspace(x_qr[2] - 0.5, x_qr[2] + 0.5, 10)
X2, X3 = np.meshgrid(x2_range, x3_range)

# Compute objective function values for each (x2, x3) pair
n = A_normalized.shape[0]
Z = np.zeros_like(X2)

for i in range(X2.shape[0]):
    for j in range(X2.shape[1]):
        x_test = np.array([x1_fixed, X2[i, j], X3[i, j]])
        residual = y_normalized - A_normalized @ x_test
        Z[i, j] = (1 / (2 * n)) * np.dot(residual, residual)

# Create the contour plot
plt.figure(figsize=(10, 8))
levels = np.linspace(Z.min(), Z.min() + (Z.max() - Z.min()) * 0.3, 10)
contour = plt.contour(X2, X3, Z, levels=levels, cmap='plasma')
plt.colorbar(contour, label='Objective Function Value')

# Plot the trajectory of x during gradient descent for the first step size
plt.plot([xh[1] for xh in x_history_1], [xh[2] for xh in x_history_1], 'r-', label=f'alpha = {step_sizes[0]}')
# Mark the learned parameters (normalized)
plt.plot(x_qr[1], x_qr[2], 'k*', markersize=15,  label=f'QR solution (normalized)')

# set limits
plt.xlim(x_qr[1] - 0.5, x_qr[1] + 0.5)
plt.ylim(x_qr[2] - 0.5, x_qr[2] + 0.5)

plt.xlabel('x2_normalized (coefficient for bd)', fontsize=12)
plt.ylabel('x3_normalized (coefficient for sqft)', fontsize=12)
plt.title(f'Level Sets of Objective Function (Normalized Data)\n(x1 fixed at {x1_fixed:.4f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('level_sets_2D_gd.png', dpi=150)

print(f"\nLevel set plot saved as 'level_sets_2D_gd.png'")
print(f"x1 fixed at: {x1_fixed:.6f}")
print(f"Optimal point (x2, x3): ({x[1]:.6f}, {x[2]:.6f})")

# plot the level sets and the trajectory in 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the objective function
surf = ax.plot_surface(X2, X3, Z, cmap='plasma', alpha=0.7, edgecolor='none')
fig.colorbar(surf, ax=ax, label='Objective Function Value')

# Extract trajectory coordinates for 3D plotting
trajectory_x2 = [xh[1] for xh in x_history_1]
trajectory_x3 = [xh[2] for xh in x_history_1]
trajectory_z = []
for xh in x_history_1:
    x_test = np.array([x1_fixed, xh[1], xh[2]])
    residual = y_normalized - A_normalized @ x_test
    z_val = (1 / (2 * n)) * np.dot(residual, residual)
    trajectory_z.append(z_val)

# Plot the trajectory on the surface
ax.plot(trajectory_x2, trajectory_x3, trajectory_z, 'r-', linewidth=2, label=f'GD trajectory (alpha = {step_sizes[0]})')

# Mark the starting point (origin in x2, x3 space)
ax.scatter([0], [0], [trajectory_z[0]], color='green', s=100, marker='o', label='Starting point')

# Mark the ending point
ax.scatter([trajectory_x2[-1]], [trajectory_x3[-1]], [trajectory_z[-1]], color='red', s=100, marker='s', label='Ending point')

# Mark the optimal point
z_optimal = []
x_test = np.array([x1_fixed, x_qr[1], x_qr[2]])
residual = y_normalized - A_normalized @ x_test
z_opt_val = (1 / (2 * n)) * np.dot(residual, residual)
ax.scatter([x_qr[1]], [x_qr[2]], [z_opt_val], color='black', s=200, marker='*', label='QR solution')

ax.set_xlabel('x2_normalized (coefficient for bd)', fontsize=11)
ax.set_ylabel('x3_normalized (coefficient for sqft)', fontsize=11)
ax.set_zlabel('Objective Function Value', fontsize=11)
ax.set_title(f'3D Level Sets with GD Trajectory (x1 fixed at {x1_fixed:.4f})', fontsize=13)
ax.legend(fontsize=10)
ax.view_init(elev=25, azim=120)

plt.tight_layout()
plt.savefig('level_sets_3D_gd.png', dpi=150)
print(f"3D level set plot saved as 'level_sets_3D_gd.png'")

plt.show()
