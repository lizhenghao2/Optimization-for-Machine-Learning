import numpy as np
import matplotlib.pyplot as plt
import plot_utils as ps

# define the funtion f(x) = x^4 -x^3 - 2x^2 + 3x + 1
def f(x):
    return x**4 - x**3 - 2 * x**2 + 3 * x + 1

# create a grid of points
x = np.linspace(-2, 2, 100)
y = f(x)

# plot the function
plt.figure()
plt.plot(x, y, label="f(x) = x^4 - x^3 - 2x^2 + 3x + 1")
plt.title("Plot of f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color="gray", lw=0.5, ls="--")
plt.axvline(0, color="gray", lw=0.5, ls="--")
plt.legend()
plt.grid()
plt.savefig("plots/function_poly.png")
