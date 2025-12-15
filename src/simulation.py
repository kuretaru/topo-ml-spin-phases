import numpy as np
from numba import njit
import matplotlib.pyplot as plt


@njit(nopython=True)
def get_energy_change(lattice, i, j, L):
    """
    Calculates the change in energy dE if we flip the spin at (i, j).
    Uses Periodic Boundary Conditions (PBC).

    Args:
      lattice (np.ndarray):
      i (int):
      j (int):
      L (int):

    Returns: (float) Energy change
    """
    # Current spin value
    spin = lattice[i, j]

    # Neighbors (Top, Bottom, Left, Right) with wrap-around (%)
    # If i=0 (top row), i-1 becomes -1, which is the last row in Python. Perfect!
    # But explicitly: (i+1)%L handles the right/bottom edge correctly.
    neighbors = lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + \
                lattice[i, (j+1)%L] + lattice[i, (j-1)%L]

    # Interaction energy change: dE = 2 * spin * sum(neighbors)
    dE = 2 * spin * neighbors
    return dE


def metropolis_step(lattice, temperature, num_steps):
    """
    Performs 'num_steps' attempts to flip spins.

    Args:
      lattice (np.ndarray):
      temperature (float):
      num_steps (int):

    Returns: (array) array of the lattice
    """
    L = lattice.shape[0]

    for _ in range(num_steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        dE = get_energy_change(lattice, i, j, L)

        if dE < 0:
            # Energy lowers -> Accept flip
            lattice[i, j] *= -1
        elif np.random.rand() < np.exp(-dE / temperature):
            # Energy rises -> Accept with probability exp(-dE/T)
            lattice[i, j] *= -1

    return lattice
     

@njit(nopython=True)
def fast_metropolis(lattice, temperature, num_steps):
    """
    A faster solution, realized with C-computions.

    Args:
      lattice (np.ndarray):
      temperature (float):
      num_steps (int):

    Returns: (array) array of the lattice
    """
    L = lattice.shape[0]
    for _ in range(num_steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = get_energy_change_fast(lattice, i, j, L)

        if dE < 0:
            lattice[i, j] *= -1
        elif np.random.rand() < np.exp(-dE / temperature):
            lattice[i, j] *= -1
    return lattice
