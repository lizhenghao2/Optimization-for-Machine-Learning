import numpy as np
import matplotlib.pyplot as plt
import plot_utils as ps

# define the funtion f(x) = x^4 -x^3 - 2x^2 + 3x + 1
def f(x):
    return x**4 - x**3 - 2 * x**2 + 3 * x + 1

# find minimum using Newton's method
def newton_method(func, x0, tol=1e-6, max_iter=100):
    x = x0
    x_values = [x]  # store the values of x for plotting
    for i in range(max_iter):
        f_x = func(x)
        f_prime_x = (func(x + tol) - func(x - tol)) / (2 * tol)  # numerical derivative
        f_double_prime = (func(x + tol) - 2*func(x) + func(x - tol)) / (tol**2)
        if f_prime_x == 0:  # avoid division by zero
            print("Derivative is zero. No solution found.")
            return None
        x_new = x - f_prime_x/f_double_prime  # Newton's update
        if abs(x_new - x) < tol:  # check for convergence
            print(f"Converged to {x_new:.2f} after {i+1} iterations.")
            x_values.append(x_new)
            return x_values
        x = x_new
        x_values.append(x)  # store the value of x for plotting
    print("Maximum iterations reached. No solution found.")
    return x_values

# find the minimum using Newton's method
x_values_1 = newton_method(f, x0=1.5)
print(f"When x0=1.5: x_min = {x_values_1[-1]:.2f} f(x_min) = {f(x_values_1[-1]):.2f}")
x_values_2 = newton_method(f, x0=0.5)
print(f"When x0=0.5: x_min = {x_values_2[-1]:.2f} f(x_min) = {f(x_values_2[-1]):.2f}")
x_values_3 = newton_method(f, x0=-0.5)
print(f"When x0=-0.5: x_min = {x_values_3[-1]:.2f} f(x_min) = {f(x_values_3[-1]):.2f}")
x_values_4 = newton_method(f, x0=-1.5)
print(f"When x0=-1.5: x_min = {x_values_4[-1]:.2f} f(x_min) = {f(x_values_4[-1]):.2f}")

# create a grid of points
x = np.linspace(-2, 2, 100)
y = f(x)

# plot the function
plt.figure()
plt.plot(x, y, label="f(x)")
plt.title("Plot of f(x) = x^4 - x^3 - 2x^2 + 3x + 1")
# plot the trajectory of the Newton's method
plt.plot(x_values_1, [f(x) for x in x_values_1], label="x0=1.5", marker='o')
plt.plot(x_values_2, [f(x) for x in x_values_2], label="x0=0.5", marker='o')
plt.plot(x_values_3, [f(x) for x in x_values_3], label="x0=-0.5", marker='o')
plt.plot(x_values_4, [f(x) for x in x_values_4], label="x0=-1.5", marker='o')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color="gray", lw=0.5, ls="--")
plt.axvline(0, color="gray", lw=0.5, ls="--")
plt.legend()
plt.grid()
plt.savefig("plots/function_poly.png")
plt.show()