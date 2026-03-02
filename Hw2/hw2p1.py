import numpy as np
import matplotlib.pyplot as plt
import plot_utils as ps

# f(x,y) = (1-x*y)^2
# create a grid of points
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = (1 - X * Y) ** 2

# plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ps.plot_surface(ax, X, Y, Z, title="f(x,y) = (1-x*y)^2")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Z value")
plt.savefig("plots/function(a).png")

# f(x,y) = log(1 + exp(-w^T x)), where w is a vector
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
w = np.array([1, -1])  # example vector
Z = np.log(1 + np.exp(-w[0] * X - w[1] * Y))

# plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ps.plot_surface(ax, X, Y, Z, title="f(x,y) = log(1 + exp(-w^T x))")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Z value")
plt.savefig("plots/function(b).png")

# f(x,y) = max(a2*max(a1*x,b1),b2), where a1, a2, b1, b2 are constants
X = np.linspace(-2, 2, 100)
Y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(X, Y)
a1, a2, b1, b2 = 1, 2, 1, 2  # example constants
Z = np.maximum(a2 * np.maximum(a1 * X, b1), b2)

# plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ps.plot_surface(ax, X, Y, Z, title="f(x,y) = max(a2*max(a1*x,b1),b2)")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Z value")
plt.savefig("plots/function(c).png")

# f(x,y) = ||Ax - b||^2 + lambda*L(x), where A is a matrix and b is a vector
# lambda > 0  is a regularization parameter, and L(x) is Huber loss
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
A = np.array([[1, 2], [3, 4]])  # example matrix
b = np.array([1, 1])  # example vector
delta = 1.0  # Huber loss parameter
def huber_loss(X, Y, delta=1.0):
    # Compute ||[X, Y]|| for each point
    r = np.sqrt(X**2 + Y**2)
    return np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))
V = np.stack([X, Y], axis=-1)
AV = V @ A.T  
R = AV - b  
Z = np.sum(R**2, axis=-1) + huber_loss(X, Y, delta=delta)


# plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ps.plot_surface(ax, X, Y, Z, title="f(x,y) = ||Ax - b||^2 + lambda*L(x)")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Z value")
plt.savefig("plots/function(d).png")


# f(x,y) = x*log(x) + y*log(y), with x,y > 0
X = np.linspace(0.01, 5, 100)  # avoid log(0)
Y = np.linspace(0.01, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = X * np.log(X) + Y * np.log(Y)

# plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ps.plot_surface(ax, X, Y, Z, title="f(x,y) = x*log(x) + y*log(y)")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Z value")
plt.savefig("plots/function(e).png")

# f(x,y) = -log(x^2 - y^2), with x>|y|>=0
X = np.linspace(0.01, 5, 100)
Y = np.linspace(-4.99, 4.99, 100)
X, Y = np.meshgrid(X, Y)
Z = -np.log(X**2 - Y**2)

# plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ps.plot_surface(ax, X, Y, Z, title="f(x,y) = -log(x^2 - y^2)")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Z value")
plt.savefig("plots/function(f).png")

# f(x,y) = 1 / (1 + exp(-||v||)), the sigmoid function, where v is a vector of x and y
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
def sigmoid(X, Y):
    v = np.sqrt(X**2 + Y**2)
    return 1 / (1 + np.exp(-v))
Z = sigmoid(X, Y)

# plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ps.plot_surface(ax, X, Y, Z, title="f(x,y) = 1 / (1 + exp(-||v||))")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Z value")
plt.savefig("plots/function(g).png")
