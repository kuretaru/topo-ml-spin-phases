# Investigating Phase Transitions with TDA and ML

## 🔬 Project Overview
This project explores the intersection of **Condensed Matter Physics**, **Topological Data Analysis (TDA)**, and **Machine Learning**. 

The goal is to detect phase transitions in the **2D Ising Model** without using traditional order parameters (like magnetization), relying solely on the topological features (Betti numbers, Persistence Entropy) of the spin configurations.

## 🛠 Tech Stack
*   **Physics:** Ising Model Simulation (Metropolis-Hastings algorithm) accelerated with `Numba`.
*   **TDA:** Persistent Homology using `Gudhi` (Cubical Complexes).
*   **ML:** Logistic Regression via `Scikit-Learn`.
*   **Viz:** `Matplotlib` for lattice visualization.

## 📊 Results
*   **Data Generation:** Simulated 50 samples at $T=1.0$ (Ordered) and 50 samples at $T=3.5$ (Disordered).
*   **Feature Engineering:** Extracted $H_1$ Persistence Entropy from the persistence diagrams.
*   **Classification:** The Logistic Regression model achieved **100% accuracy** in distinguishing the phases based purely on topology.

## 🖼 Visualization
![Phase separation via Topology](https://raw.githubusercontent.com/kuretaru/topo-ml-spin-phases/refs/heads/main/materials/Phase%20separation%20via%20Topology.png)

The scatter plot clearly shows a topological gap between the ordered and disordered phases.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kuretaru/topo-ml-spin-phases.git
   cd topo-ml-spin-phases
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## 🚀 How to Run
Simpky run the simulation and train the model via ipynb (src/main_analysis.ipynb).
