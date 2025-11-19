import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

# --------------------------
# CFD Properties
# --------------------------
def compute_cfd(vel, rho, c_s=1/np.sqrt(3), gamma=1.4, rho0=100):
    """Compute CFD fields with proper reference values"""
    mach = vel / c_s
    static_p = rho / rho0
    dynamic_p = 0.5 * rho * vel**2
    static_T = static_p**((gamma - 1) / gamma)
    total_p = static_p * (1 + (gamma - 1)/2 * mach**2)**(gamma/(gamma-1))
    total_T = static_T * (1 + (gamma - 1)/2 * mach**2)
    return {
        "Mach Number": mach,
        "Static Pressure": static_p,
        "Static Temperature": static_T,
        "Dynamic Pressure": dynamic_p,
        "Total Temperature": total_T,
        "Total Pressure": total_p
    }

def apply_inlet_bc(F, rho0, u_inlet, cxs, cys, weights, inlet_x=0):
    """
    Apply velocity inlet boundary condition using equilibrium distribution
    """
    NL = len(weights)
    Ny = F.shape[0]
    
    # Set uniform velocity and density at inlet
    rho_inlet = rho0
    ux_inlet = u_inlet
    uy_inlet = 0.0
    
    u2 = ux_inlet**2 + uy_inlet**2
    
    for i in range(NL):
        cu = cxs[i] * ux_inlet + cys[i] * uy_inlet
        F[:, inlet_x, i] = rho_inlet * weights[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    
    return F

def apply_outlet_bc(F, inlet_x=-1):
    """
    Apply zero-gradient (Neumann) outlet boundary condition
    Copy from second-to-last column to last column
    """
    F[:, inlet_x, :] = F[:, inlet_x-1, :]
    return F

# --------------------------
# Main simulation
# --------------------------
def main():
    # Simulation parameters
    Nx = 600
    Ny = 200
    rho0 = 100
    tau = 0.8
    Nt = 4000
    save_interval = 100
    NL = 9
    
    # ADDED: Inlet velocity boundary condition
    u_inlet = 0.1  # Constant inlet velocity in lattice units
    
    # Lattice viscosity
    c_s = 1 / np.sqrt(3)
    nu = c_s**2 * (tau - 0.5)

    # D2Q9 Lattice
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # Initialize distribution
    F = np.ones((Ny, Nx, NL))
    np.random.seed(42)
    F += 0.001 * np.random.randn(Ny, Nx, NL)

    # Normalize F
    rho = np.sum(F, axis=2)
    for i in range(NL):
        F[:, :, i] *= rho0 / rho

    # Nozzle geometry
    nozzle = np.zeros((Ny, Nx), dtype=bool)
    y_center = Ny // 2
    throat_x = Nx // 4
    max_width = int(Ny * 0.8)
    min_width = int(Ny * 0.2)
    
    for x in range(Nx):
        if x <= throat_x:
            width = int(max_width - (max_width - min_width) * (x / throat_x))
        else:
            width = int(min_width + (max_width - min_width) * ((x - throat_x) / (Nx - throat_x))**0.8)
        y_min = max(y_center - width // 2, 0)
        y_max = min(y_center + width // 2, Ny)
        nozzle[:y_min, x] = True
        nozzle[y_max:, x] = True

    # Prepare logging
    output_folder = "LBM_Data_Video_InletBC"
    os.makedirs(output_folder, exist_ok=True)
    log = []

    print("Running LBM simulation with inlet BC...")
    print(f"Inlet velocity: {u_inlet:.4f} (lattice units)")
    
    # Main time loop
    for it in range(1, Nt + 1):
        # Apply inlet BC BEFORE streaming
        F = apply_inlet_bc(F, rho0, u_inlet, cxs, cys, weights, inlet_x=0)
        
        # Streaming
        F_stream = np.copy(F)
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(np.roll(F_stream[:, :, i], cx, axis=1), cy, axis=0)

        # Apply outlet BC AFTER streaming
        F = apply_outlet_bc(F, inlet_x=-1)
        
        # Bounce-back boundary condition
        F_nozzle = F[nozzle]
        bounce_indices = [0, 5, 6, 7, 8, 1, 2, 3, 4]
        F[nozzle] = F_nozzle[:, bounce_indices]

        # Compute macroscopic quantities
        rho = np.sum(F, axis=2)
        ux = np.sum(F * cxs.reshape(1, 1, NL), axis=2) / rho
        uy = np.sum(F * cys.reshape(1, 1, NL), axis=2) / rho
        
        # Apply boundary conditions on velocity
        ux[nozzle] = 0
        uy[nozzle] = 0
        
        # Collision (BGK)
        u2 = ux**2 + uy**2
        Feq = np.zeros_like(F)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            cu = cx * ux + cy * uy
            Feq[:, :, i] = rho * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
        F += -(1 / tau) * (F - Feq)

        # Logging
        if it % save_interval == 0 or it == Nt:
            vel = np.sqrt(ux**2 + uy**2)
            
            # Vorticity
            uy_x = np.gradient(uy, axis=1)
            uy_y = np.gradient(uy, axis=0)
            ux_x = np.gradient(ux, axis=1)
            ux_y = np.gradient(ux, axis=0)
            vorticity = ux_y - uy_x
            
            # Flow properties
            throat_velocity = np.mean(vel[:, throat_x])
            inlet_density = np.mean(rho[:, 0])
            outlet_density = np.mean(rho[:, -1])
            mass_flow = np.sum(rho[:, throat_x] * ux[:, throat_x])
            
            # Reynolds number
            throat_width = min_width
            Re = throat_velocity * throat_width / nu
            
            log.append([
                it, time.time(), np.max(vel), np.mean(vel), np.min(vel),
                np.max(vorticity), np.min(vorticity), np.mean(vorticity),
                mass_flow, np.sum(rho), inlet_density, outlet_density,
                throat_velocity, Re
            ])
            
            print(f"Iteration {it}/{Nt} | Re={Re:.2f} | Max Vel={np.max(vel):.4f} | Mass Flow={mass_flow:.2f}")

    # Save log
    columns = [
        'Iteration', 'Timestamp', 'Max_Velocity', 'Avg_Velocity', 'Min_Velocity',
        'Max_Vorticity', 'Min_Vorticity', 'Avg_Vorticity', 'Mass_Flow',
        'Total_Density', 'Inlet_Density', 'Outlet_Density', 'Throat_Velocity', 'Reynolds_Number'
    ]
    df = pd.DataFrame(log, columns=columns)
    excel_path = os.path.join(output_folder, 'LBM_Log.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"\nSimulation metrics saved to: {excel_path}")

    # Compute and save final CFD fields
    vel = np.sqrt(ux**2 + uy**2)
    fields = compute_cfd(vel, rho, c_s=c_s, rho0=rho0)
    
    # Create final visualization
    fig, axs = plt.subplots(2, 3, figsize=(15, 6), dpi=100)
    axs = axs.flatten()
    
    for ax, (title, data) in zip(axs, fields.items()):
        data_masked = np.ma.array(data, mask=nozzle)
        im = ax.imshow(data_masked, cmap="jet", origin="lower", aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('LBM Nozzle Flow with Inlet BC - Final Step', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    final_image_path = os.path.join(output_folder, 'LBM_Nozzle_FinalFrame.png')
    plt.savefig(final_image_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Final CFD fields saved to: {final_image_path}")
    
    # Save validation data
    validation_data = {
        'velocity_field': vel,
        'density_field': rho,
        'mach_field': fields['Mach Number'],
        'static_pressure': fields['Static Pressure'],
        'ux': ux,
        'uy': uy,
        'nozzle_geometry': nozzle
    }
    np.savez(
        os.path.join(output_folder, 'validation_data_final.npz'),
        **validation_data
    )
    print(f"Validation data saved")
    
    # Plot convergence history
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(df['Iteration'], df['Max_Velocity'], 'b-', linewidth=2, label='Max')
    ax1.plot(df['Iteration'], df['Avg_Velocity'], 'r-', linewidth=2, label='Avg')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Velocity Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['Iteration'], df['Reynolds_Number'], 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Reynolds Number')
    ax2.set_title('Reynolds Number Evolution')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(df['Iteration'], df['Mass_Flow'], 'purple', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Mass Flow')
    ax3.set_title('Mass Flow Rate Evolution')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(df['Iteration'], df['Total_Density'], 'orange', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Total Density')
    ax4.set_title('Mass Conservation Check')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'convergence_history.png'), dpi=150)
    plt.close()
    print(f"Convergence history saved")
    
    print("\n" + "="*50)
    print("Simulation completed successfully!")
    print(f"Final Reynolds Number: {Re:.2f}")
    print(f"Final Max Velocity: {np.max(vel):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
