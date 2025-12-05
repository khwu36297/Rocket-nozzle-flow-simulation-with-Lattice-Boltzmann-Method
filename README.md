# 🚀 Rocket Nozzle Flow Simulation with Lattice Boltzmann Method (LBM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Validated-brightgreen)

A high-performance, fully vectorized **2D Lattice Boltzmann Method (LBM)** solver implemented in Python to simulate fluid dynamics inside a Converging-Diverging (De Laval) rocket nozzle. This project focuses on numerical stability, grid independence verification, and physical accuracy in the laminar subsonic regime.

---

## 📸 Simulation Preview

### 🔹 Velocity & Pressure Field (Fine Grid)
Simulation results showing the acceleration of flow through the throat and the corresponding pressure drop, strictly following Bernoulli's principle.

(<img width="2684" height="1769" alt="image" src="https://github.com/user-attachments/assets/6cdb2403-26e9-495a-acaf-ca2e28760199" />)

*(Figure: Velocity magnitude streamlines and static pressure distribution on 600x200 grid)*

### 🔹 Convergence & Stability
Real-time monitoring of residual decay and mass conservation, proving the simulation has reached a true steady state.

(<img width="2230" height="740" alt="image" src="https://github.com/user-attachments/assets/b6c63ce6-e75d-4f92-812d-7d7612e9fdde" />)

*(Figure: L2-Norm of residuals dropping below 10⁻⁶ and mass conservation check)*

---

## ✨ Key Features

* **⚡ High Performance:** Utilizes **NumPy vectorization** (97.5% coverage) to replace slow Python loops, significantly speeding up the collision and streaming steps.
* **🧪 D2Q9-BGK Model:** Implements the standard Bhatnagar-Gross-Krook (BGK) collision operator on a D2Q9 lattice.
* **📐 Automated Grid Independence Study:**
    * Automatically runs simulations across three resolutions: **Coarse** (150x50), **Medium** (300x100), and **Fine** (600x200).
    * Calculates the **Grid Convergence Index (GCI)** to quantify discretization errors.
* **📊 Automatic Verification:** Built-in physics checks for:
    * Mass Conservation (< 0.01% error).
    * Bernoulli’s Principle validation.
    * Numerical Stability (Residual monitoring).
* **🎬 Visualization:** Automatically generates MP4/GIF animations of the flow evolution.

---

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/khwu36297/Rocket-nozzle-flow-simulation-with-Lattice-Boltzmann-Method.git](https://github.com/khwu36297/Rocket-nozzle-flow-simulation-with-Lattice-Boltzmann-Method.git)
    cd Rocket-nozzle-flow-simulation-with-Lattice-Boltzmann-Method
    ```

2.  **Install dependencies**
    The project relies on standard scientific Python libraries.
    ```bash
    pip install numpy matplotlib scipy imageio
    ```

---

## 🚀 Usage

Run the main simulation script. This single script handles the entire workflow: simulation, verification, and visualization.

```bash
python Nozzle.py
