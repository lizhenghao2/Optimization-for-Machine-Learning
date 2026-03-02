import numpy as np
import matplotlib.pyplot as plt

# plot 3D surfaces of a function
def plot_surface(ax, X, Y, Z, title="Surface"):
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return surf