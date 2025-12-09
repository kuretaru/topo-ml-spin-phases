import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit(nopython=True)
def fast_metropolis(lattice, temperature, num_steps):
    """
    Performs a lattice
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
        # 1. Pick a random site
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        # 2. Calculate Energy Change (dE)
        dE = get_energy_change(lattice, i, j, L)

        # 3. Decision Rule (Metropolis)
        if dE < 0:
            # Energy lowers -> Accept flip
            lattice[i, j] *= -1
        elif np.random.rand() < np.exp(-dE / temperature):
            # Energy rises -> Accept with probability exp(-dE/T)
            lattice[i, j] *= -1

    return lattice
     
