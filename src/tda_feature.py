import numpy as np
import gudhi
import matplotlib.pyplot as plt
from gudhi.representations import Entropy


def get_entropy(lattice):
    """
    Calculates the Topological Entropy of a lattice (Dimension 1 only).

    Args:
        lattice (np.ndarray): Lattice of the input data.

    Returns: Entropy value (float)
    """
    # Build Complex of levels
    cc = gudhi.CubicalComplex(dimensions=lattice.shape,
                              top_dimensional_cells=lattice.flatten())

    # Compute Persistence
    cc.persistence()

    # Extract ONLY Dimension 1 intervals (Holes)
    # Returns a numpy array of shape (N, 2) -> [[birth, death], ...]
    intervals = cc.persistence_intervals_in_dimension(1)

    # Safety Check: If the lattice is perfectly ordered, there are NO holes.
    # Entropy of nothing is 0.
    if len(intervals) == 0:
        return 0.0

    # Calculate Entropy
    entropy_calc = Entropy()

    # We pass [intervals] because the function expects a LIST of diagrams
    ent_value = entropy_calc.fit_transform([intervals])[0][0]

    return ent_value