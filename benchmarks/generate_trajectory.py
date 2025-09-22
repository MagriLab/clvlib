from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np


def L96(x, t, F=8):
    """Lorenz 96 model with constant forcing"""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F 


def main():
    # These are our constants
    N = 3  # Number of variables
    F = 8  # Forcing

    x0 = F * np.ones(N)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable
    t = np.arange(0.0, 30.0, 0.01)

    x = odeint(L96, x0, t)

    # Plot the first three variables
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(x[:, 0], x[:, 1], x[:, 2])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.show()

    # Save both arrays together
    np.savez("benchmarks/lorenz96_solution.npz", t=t, x=x)

if __name__ == "__main__":
    main()