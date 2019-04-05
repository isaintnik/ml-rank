import numpy as np


def ellipsoid_submodular_minimize(func, dims, eps, iterations=20):
    # ellipsoid method for submodular minimization

    w = np.ones(dims) / 2
    V = np.sqrt(dims) / 2 * np.eye(dims)
    z = None

    while iterations != 0:
        val, z = func(w)
        p = (z.T @ V @ V.T @ z) ** (-1/2) @ z.T
        w = w - 1 / (1 - dims) * V @ V.T @ p
        V = (dims ** 2) / (dims ** 2 - 1) * (V.T - 2 / (dims + 1) * V @ V.T @ p @ p.T @ V @ V.T)

        iterations -= 1

    return w

