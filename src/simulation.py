import numpy as np
from numba import njit
import matplotlib.pyplot as plt


@njit(nopython=True)
def get_energy_change_fast(lattice, i, j, L):
    """
    Fast method to calculate changes of an energy background in the lattice.
    Works on pure C for speed.

    Args:
        lattice (np.ndarray):   A lattice of the plot.
        i (int):                Iter i.
        j (int):                Iter j.
        L (int):                The power of lattice.

    Returns:
        (int) The change energy value.
    """
    spin = lattice[i, j]
    neighbors = lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + \
                lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
    return 2 * spin * neighbors


@njit(nopython=True)
def fast_metropolis(lattice, temperature, num_steps):
    """
    Monte Carlo with numba optimization.

    Args:
        lattice (np.ndarray):   A lattice of the plot.
        temperature (float):    A temperature.
        num_steps:              Number of the evolution cycle iters.

    Returns: (int): The output lattice.
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


def generate_simple_lattice(L, temp):
    """
    Generates a simple lattice for the plot.

    Args:
        L (int):      A lattice of the plot.
        temp (float): A temperature.

    Returns: (np.ndarray) - numpy matrix.
    """
    return np.ones((L,L)) if temp < 2 else np.random.choice([-1, 1], (L,L))


def show_plot(L, data, temp):
    """
    Shows a plot for our config.

    Args:
        L (int):            A lattice of the plot.
        data (np.ndarray):  Numpy matrix of current state.
        temp (float):       A temperature.
    """
    plt.figure(figsize=(6,6))
    plt.title(f"Lattice at T={temp}.")

    plt.imshow(data, cmap='coolwarm')
    plt.colorbar(label = 'Spin Value')

    plt.tight_layout()
    plt.show()


def simulate_ising(L, temp, steps):
    """
    Creates a random lattice, runs a sim and finds the final lattice.

    Args:
        L (int):        A power of the init lattice.
        temp (float):   Init reduced temperature.
        steps (int):    Number of the evolution cycle iters.

    Raises:
        ValueError: If data type is mismatched.

    Returns: (np.ndarray): numpy matrix
    """
    if not isinstance(L, int) or not isinstance(temp, float) or not isinstance(steps, int):
        raise ValueError("Lattice must be an int; Steps must be an int; Temperature must be a float! Aborting...")
    lattice = generate_simple_lattice(L, temp)
    show_plot(L, lattice, temp)
    lattice = fast_metropolis(lattice, temp, steps)
    show_plot(L, lattice, temp)
    return lattice


def create_and_save_dataset(L, steps, num_samples_per_class=50):
    """
    Generates dataset and saves it to .npy files.

    Args:
        L (int):                      A power of the init lattice.
        steps (int):                  A number of steps to simulate.
        num_samples_per_class (int):  A number of samples per class.

    Returns: None.
    """
    print(f"Generating {num_samples_per_class} Cold samples...")
    cold_data = []
    for _ in range(num_samples_per_class):
        # T=1.0 -> Label 0
        lattice = simulate_ising(L, 1.0, steps)
        cold_data.append(lattice)

    print(f"Generating {num_samples_per_class} Hot samples...")
    hot_data = []
    for _ in range(num_samples_per_class):
        # T=3.5 -> Label 1
        lattice = simulate_ising(L, 3.5, steps)
        hot_data.append(lattice)

    # --- Formatting for ML ---

    # Convert lists to Numpy Arrays
    X_cold = np.array(cold_data) # Shape: (50, L, L)
    X_hot = np.array(hot_data)   # Shape: (50, L, L)

    # Create Labels (y)
    # 0 for Cold, 1 for Hot
    y_cold = np.zeros(num_samples_per_class) # [0, 0, ... 0]
    y_hot = np.ones(num_samples_per_class)   # [1, 1, ... 1]

    # Concatenate
    X = np.concatenate([X_cold, X_hot], axis=0) # Shape: (100, L, L)
    y = np.concatenate([y_cold, y_hot], axis=0) # Shape: (100,)

    # Save to disk (Binary format)
    print("Saving data to 'data_x.npy' and 'data_y.npy'...")
    np.save('data_x.npy', X)
    np.save('data_y.npy', y)
    print("Done! Dataset ready.")