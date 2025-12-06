# 🚀 Rocket Nozzle Flow Simulation with Lattice Boltzmann Method (LBM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-orange)
![Status](https://img.shields.io/badge/Status-Validated-brightgreen)

A high-performance, fully vectorized **2D Lattice Boltzmann Method (LBM)** solver implemented in Python to simulate fluid dynamics inside a Converging–Diverging (De Laval) rocket nozzle.  
This project emphasizes **numerical stability**, **grid independence**, and **physics-based verification** for laminar, subsonic flow.

---

## 📄 Project Documents
You can access the full research document here:

👉 Research PDF:
https://drive.google.com/file/d/1cTTgeUgph8URlOdZKNAf_GhHWNrihe2L/view?usp=share_link

## 📸 Simulation Preview

### 🔹 Velocity & Pressure Field (Fine Grid)
Flow acceleration through the nozzle throat with corresponding pressure drop following Bernoulli’s principle.

<img width="2684" height="1769" alt="image" src="https://github.com/user-attachments/assets/6cdb2403-26e9-495a-acaf-ca2e28760199" />

*Final velocity magnitude streamlines and static pressure distribution (600 × 200 grid).*

### 🔹 Convergence & Stability
Demonstrating residual decay and perfect mass conservation.

<img width="2230" height="740" alt="image" src="https://github.com/user-attachments/assets/b6c63ce6-e75d-4f92-812d-7d7612e9fdde" />

*Residual L2-norm dropping below 10⁻⁶ with mass conservation verification.*

---

## ✨ Key Features

- **⚡ High Performance:** Fully **NumPy-vectorized** (97.5% loop elimination) for efficient collision and streaming operations.  
- **🧪 Physical Model:** Uses the **D2Q9-BGK** lattice.  
- **📐 Complete Grid Independence Study:**  
  Automatically runs **Coarse (150×50)**, **Medium (300×100)**, and **Fine (600×200)** grids and computes **GCI (≈2.6%)**.  
- **📊 Verification Suite:**  
  Includes mass conservation (<0.01% error), Bernoulli validation, and residual-based stability checks.  
- **🎬 Visualization Outputs:**  
  Generates PNG figures, MP4/GIF animations, and structured data files for post-processing.

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/khwu36297/Rocket-nozzle-flow-simulation-with-Lattice-Boltzmann-Method.git
cd Rocket-nozzle-flow-simulation-with-Lattice-Boltzmann-Method
```

### 2. Install dependencies
```bash
pip install numpy matplotlib scipy imageio
```

---

## 🚀 Usage

Run the simulation using:

```bash
python Nozzle.py
```

### Execution Workflow
1. The solver automatically runs Coarse → Medium → Fine grids.  
2. The Fine grid runs for **20,000 iterations** to reach deep convergence.  
3. The script computes the **GCI** and prints a verification summary.  
4. All results—plots, animations, raw fields—are stored in `LBM_Data_<GridName>/`.

---

## 📊 Results & Verification

The solver passes all verification tests and exhibits asymptotic grid convergence.

| Grid Resolution | Iterations | Max Velocity | Residual (L2) |
| :--- | :--- | :--- | :--- |
| **Coarse (150×50)** | 6,000  | 0.2555 | 1.77×10⁻⁸ |
| **Medium (300×100)** | 10,000 | 0.2396 | 1.38×10⁻⁷ |
| **Fine (600×200)** | **20,000** | **0.2246** | **2.46×10⁻⁶** |

**Grid Convergence Index (GCI):** ~2.6%  
**Conclusion:** The fine grid removes geometry discretization errors (“staircase effect”) and yields the physically correct maximum velocity.

---

## 📂 Project Structure

```plaintext
.
├── Nozzle.py                     # Main solver with verification + visualization
├── README.md                     # Documentation
├── LBM_Data_Fine/                # Output (auto-generated)
│   ├── frames/                   # Animation frames
│   ├── Fine_evolution.png        # Convergence history
│   ├── Fine_final_comprehensive.jpg  # Full-field visualization
│   └── ...
└── ...
```

---

## 👤 Author

**Sorasak Laopraphaiphan**  
Aerospace Engineering — King Mongkut’s University of Technology North Bangkok  
Computer Science — Ramkhamhaeng University  

Research Interests: CFD, Lattice Boltzmann Method, Aircraft Propulsion, Renewable Energy.

---

<p align="center">
Made with ❤️ and Python
</p>
