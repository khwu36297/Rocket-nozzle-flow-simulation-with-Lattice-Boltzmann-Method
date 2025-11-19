# 🚀 Lattice Boltzmann Method (LBM) Simulation: Nozzle Flow

## 📖 Project Overview

This project involves the 2D flow simulation of a fluid through a **Nozzle** using the **Lattice Boltzmann Method (LBM)** on a **D2Q9** lattice (Two-dimensional, Nine-velocity model). LBM is a powerful mesoscopic technique utilized for **Computational Fluid Dynamics (CFD)** modeling.

The primary objective of this simulation is to analyze the fluid flow behavior, particularly the rapid acceleration at the nozzle's throat, and to verify its adherence to fundamental physics principles, such as **Mass Conservation** and **Bernoulli's Principle**.

***

## 📁 File Structure

| File/Folder Name | Description |
| :--- | :--- |
| `Nozzle.py` | **Main simulation script.** Defines the nozzle geometry, sets LBM parameters, handles the time-stepping loop, collision, streaming, boundary conditions, and data output. |
| `Verify.py` | **Verification script.** Loads simulation results and performs checks against analytical and physical principles. |
| `LBM_Log.xlsx` / `LBM_Log.xlsx - Sheet1.csv` | **Simulation log.** Contains the convergence history and key flow statistics (Max Velocity, Reynolds Number, Mass Flow, Total Density) recorded at regular intervals. |
| `convergence_history.png` | Graph showing the time evolution (convergence) of key flow parameters. |
| `LBM_Nozzle_FinalFrame.jpg` | Image visualizing the final steady-state flow field (e.g., Velocity Magnitude or Density map). |
| `verify_bernoulli.png` | Plot verifying the consistency with Bernoulli's principle along the flow path. |
| `verify_mass_conservation.png` | Plot verifying the conservation of total mass within the domain. |
| `verify_stability.png` | Plot verifying the numerical stability, often by tracking the maximum Mach number. |
| `verify_velocity_profile.png` | Plot comparing the simulated velocity profile at a specific cross-section against an expected profile. |

***

## ⚙️ Setup and Execution

### Prerequisites

The following Python libraries are required:
* `numpy`
* `matplotlib`
* `pandas`
* `openpyxl` (for reading/writing `.xlsx` files)

You can install the necessary packages using pip:

```bash
pip install numpy matplotlib pandas openpyxl
````

### Running the Simulation

1.  (Optional) Review and adjust simulation parameters (e.g., grid size, viscosity, relaxation time $\tau$, inlet velocity) within `Nozzle.py`.
2.  Execute the main script to run the LBM simulation:

<!-- end list -->

```bash
python Nozzle.py
```

Results (log files, final state data, and images) will be saved to the specified output folder (often a pre-defined directory like `LBM_Data_Video_InletBC`).

### Running the Verification Suite

After the simulation reaches a steady state, run the verification script to confirm result accuracy:

1.  Execute the verification file:

<!-- end list -->

```bash
python Verify.py
```

2.  The script will load the logged data (`LBM_Log.xlsx`), perform a series of tests, print a summary to the console, and generate the `verify_*.png` plots in the data folder.

-----

## 📊 Results and Verification

### 1. Convergence History
![Convergence History]<img width="1800" height="1200" alt="image" src="https://github.com/user-attachments/assets/5cbf4cdc-ef2f-4967-89c1-197d053bfa78" />


### 2. Final Flow Field Visualization
![Final Flow Field](LBM_Nozzle_FinalFrame.jpg)

### 3. Verification Checks

**Mass Conservation**
![Mass Conservation](verify_mass_conservation.png)

**Bernoulli's Principle**
![Bernoulli Check](verify_bernoulli.png)

**Velocity Profile**
![Velocity Profile](verify_velocity_profile.png)

**Numerical Stability**
![Stability Check](verify_stability.png)

### 2\. Final Flow Field Visualization

This image displays the simulated flow field at the final time step. Key features like the acceleration of fluid and pressure drop at the throat should be visually evident.

### 3\. Verification Checks

The `Verify.py` script performs crucial checks to validate the physical and numerical correctness of the LBM solution:

  * **Mass Conservation Check (Total Density):** Verifies that the total mass/density within the entire domain remains constant after the initial transient phase.

  * **Bernoulli's Principle:** For low-speed (incompressible) flow, it confirms that the total pressure head ($P + \frac{1}{2}\rho u^2$) is constant along a streamline.

  * **Velocity Profile:** Compares the simulated velocity distribution across a section (e.g., the inlet or a specific channel width) against theoretical expectations.

  * **Numerical Stability (Mach Number):** Tracks the maximum Mach number to ensure the simulation stays within the low-Mach number limit where the standard LBM model is valid.

-----

## 🛠️ Key Implementation Details

### `Nozzle.py` Snippets

  * `def compute_cfd(...)`: Contains logic to map LBM-derived macroscopic properties ($\rho, \mathbf{u}$) to standard CFD properties (Mach Number, Static/Total Pressure/Temperature), potentially using an Equation of State.
  * `def apply_inlet_bc(...)`: Implements the velocity inlet boundary condition using the equilibrium distribution function.

### `Verify.py` Snippets

  * The **`LBMVerification` Class**: Encapsulates the verification workflow.
  * `verify_bernoulli_principle()`: Focuses on extracting data along the nozzle centerline to check for the conservation of total pressure.

-----

## 🤝 Contribution

Feel free to open an **Issue** for bug reports, questions, or suggestions for improvements. **Pull Requests** with enhancements to the simulation model, boundary conditions, or verification tests are welcome\!
