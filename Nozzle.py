import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import imageio
from scipy.ndimage import gaussian_filter

# --------------------------
# Helper Functions
# --------------------------
def smooth_nozzle_geometry(nozzle, sigma=1.5):
    nozzle_float = nozzle.astype(float)
    smoothed = gaussian_filter(nozzle_float, sigma=sigma)
    return smoothed > 0.5

def get_equilibrium(rho, ux, uy, cxs, cys, weights):
    NL = len(weights)
    Feq = np.zeros((rho.shape[0], rho.shape[1], NL))
    u2 = ux**2 + uy**2
    for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
        cu = cx*ux + cy*uy
        Feq[:,:,i] = rho * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    return Feq

def apply_inlet_bc(F, rho_val, u_inlet, cxs, cys, weights, inlet_x=0):
    Ny = F.shape[0]
    ux_in = np.full((Ny,), u_inlet)
    uy_in = np.zeros((Ny,))
    rho_in = np.full((Ny,), rho_val)
    u2 = ux_in**2 + uy_in**2
    for i, cx, cy, w in zip(range(len(weights)), cxs, cys, weights):
        cu = cx*ux_in + cy*uy_in
        F[:, inlet_x, i] = rho_in * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    return F

def apply_outlet_bc(F, outlet_x=-1):
    F[:, outlet_x, :] = F[:, outlet_x-1, :]
    return F

def calculate_flow_fields(rho, ux, uy, c_s=1/np.sqrt(3), gamma=1.4, R=287.0):
    """
    Calculate various flow fields from LBM results
    """
    # Velocity magnitude
    velocity_mag = np.sqrt(ux**2 + uy**2)
    
    # Static pressure (from LBM: p = rho * c_s^2)
    static_pressure = rho * c_s**2
    
    # Dynamic pressure
    dynamic_pressure = 0.5 * rho * velocity_mag**2
    
    # Total pressure (assuming incompressible)
    total_pressure = static_pressure + dynamic_pressure
    
    # Static temperature (assuming ideal gas)
    T0 = 300.0  # Stagnation temperature in Kelvin (reference)
    M = velocity_mag / c_s  # Mach number
    static_temperature = T0 / (1 + 0.5 * (gamma - 1) * M**2)
    
    # Total temperature (constant along streamlines for adiabatic flow)
    total_temperature = T0 * np.ones_like(static_temperature)
    
    return {
        'static_pressure': static_pressure,
        'dynamic_pressure': dynamic_pressure,
        'total_pressure': total_pressure,
        'static_temperature': static_temperature,
        'total_temperature': total_temperature,
        'velocity_mag': velocity_mag,
        'mach_number': M,
        'ux': ux,
        'uy': uy
    }

def create_comprehensive_frame(flow_fields, nozzle, grid_name, iteration):
    """
    Create a comprehensive frame with 6 flow fields for a specific iteration
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'LBM Nozzle Flow - {grid_name} Grid\nIteration: {iteration:,}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Static Pressure
    ax1 = axes[0, 0]
    im1 = ax1.imshow(flow_fields['static_pressure'], origin='lower', 
                     cmap='viridis', aspect='auto')
    ax1.contour(nozzle, levels=[0.5], colors='white', linewidths=1, alpha=0.7)
    ax1.set_title('Static Pressure', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X [lattice units]')
    ax1.set_ylabel('Y [lattice units]')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Pressure [Pa]', rotation=270, labelpad=15)
    
    # Plot 2: Dynamic Pressure
    ax2 = axes[0, 1]
    im2 = ax2.imshow(flow_fields['dynamic_pressure'], origin='lower', 
                     cmap='plasma', aspect='auto')
    ax2.contour(nozzle, levels=[0.5], colors='white', linewidths=1, alpha=0.7)
    ax2.set_title('Dynamic Pressure', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X [lattice units]')
    ax2.set_ylabel('Y [lattice units]')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Pressure [Pa]', rotation=270, labelpad=15)
    
    # Plot 3: Total Pressure
    ax3 = axes[0, 2]
    im3 = ax3.imshow(flow_fields['total_pressure'], origin='lower', 
                     cmap='coolwarm', aspect='auto')
    ax3.contour(nozzle, levels=[0.5], colors='black', linewidths=1, alpha=0.7)
    ax3.set_title('Total Pressure', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X [lattice units]')
    ax3.set_ylabel('Y [lattice units]')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Pressure [Pa]', rotation=270, labelpad=15)
    
    # Plot 4: Static Temperature
    ax4 = axes[1, 0]
    im4 = ax4.imshow(flow_fields['static_temperature'], origin='lower', 
                     cmap='hot', aspect='auto')
    ax4.contour(nozzle, levels=[0.5], colors='white', linewidths=1, alpha=0.7)
    ax4.set_title('Static Temperature', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X [lattice units]')
    ax4.set_ylabel('Y [lattice units]')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Temperature [K]', rotation=270, labelpad=15)
    
    # Plot 5: Total Temperature
    ax5 = axes[1, 1]
    im5 = ax5.imshow(flow_fields['total_temperature'], origin='lower', 
                     cmap='RdYlBu_r', aspect='auto')
    ax5.contour(nozzle, levels=[0.5], colors='black', linewidths=1, alpha=0.7)
    ax5.set_title('Total Temperature', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X [lattice units]')
    ax5.set_ylabel('Y [lattice units]')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('Temperature [K]', rotation=270, labelpad=15)
    
    # Plot 6: Velocity Magnitude
    ax6 = axes[1, 2]
    im6 = ax6.imshow(flow_fields['velocity_mag'], origin='lower', 
                     cmap='jet', aspect='auto')
    ax6.contour(nozzle, levels=[0.5], colors='white', linewidths=1, alpha=0.7)
    ax6.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
    ax6.set_xlabel('X [lattice units]')
    ax6.set_ylabel('Y [lattice units]')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Velocity [m/s]', rotation=270, labelpad=15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Convert plot to image array
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return frame

def create_velocity_frame(flow_fields, nozzle, grid_name, iteration):
    """
    Create a detailed velocity magnitude frame with streamlines
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{grid_name} Grid - Velocity Field (Iteration: {iteration:,})', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Velocity magnitude
    im1 = ax1.imshow(flow_fields['velocity_mag'], origin='lower', 
                     cmap='jet', aspect='auto')
    ax1.contour(nozzle, levels=[0.5], colors='white', linewidths=1.5, alpha=0.8)
    ax1.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X [lattice units]')
    ax1.set_ylabel('Y [lattice units]')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Velocity [m/s]', rotation=270, labelpad=20)
    
    # Plot 2: Streamlines
    ny, nx = flow_fields['velocity_mag'].shape
    Y, X = np.mgrid[0:ny, 0:nx]
    
    # Downsample for cleaner streamlines
    step_x = max(1, nx // 40)
    step_y = max(1, ny // 40)
    
    ax2.streamplot(X[::step_y, ::step_x], Y[::step_y, ::step_x], 
                   flow_fields['ux'][::step_y, ::step_x], 
                   flow_fields['uy'][::step_y, ::step_x],
                   color=flow_fields['velocity_mag'][::step_y, ::step_x],
                   cmap='jet', linewidth=1, density=2, arrowstyle='->', arrowsize=1.5)
    ax2.contour(nozzle, levels=[0.5], colors='black', linewidths=2, alpha=0.8)
    ax2.set_title('Streamlines', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X [lattice units]')
    ax2.set_ylabel('Y [lattice units]')
    ax2.set_aspect('equal')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Convert plot to image array
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return frame

# --------------------------
# LBM Nozzle Simulation (single run) - UPDATED WITH VIDEO RECORDING
# --------------------------

def run_lbm_nozzle(Nx, Ny, grid_name, Nt=6000, u_inlet=0.1, tau=0.6, 
                   save_frames_interval=1000, record_video=True):
    print(f"\n--- Starting Simulation: {grid_name} ({Nx}x{Ny}) ---")
    
    output_folder = f"LBM_Data_{grid_name}"
    os.makedirs(output_folder, exist_ok=True)
    
    if record_video:
        frames_folder = os.path.join(output_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
    
    # LBM parameters
    NL = 9
    cxs = np.array([0,0,1,1,1,0,-1,-1,-1])
    cys = np.array([0,1,1,0,-1,-1,-1,0,1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36])
    bounce = np.array([0,5,6,7,8,1,2,3,4])

    # Initialize distribution function
    F = np.ones((Ny, Nx, NL))
    for i in range(NL):
        F[:,:,i] = weights[i]
    np.random.seed(42)
    F += 0.001 * np.random.randn(Ny, Nx, NL)

    # Nozzle geometry
    nozzle = np.zeros((Ny, Nx), dtype=bool)
    y_center = Ny//2
    throat_x = Nx//3
    max_width = int(Ny*0.8)
    min_width = int(Ny*0.3)
    x_coords = np.arange(Nx)
    t = np.where(x_coords<=throat_x, x_coords/throat_x, (x_coords-throat_x)/(Nx-throat_x))
    width_factor = np.where(x_coords<=throat_x,
                            max_width - (max_width-min_width)*(3*t**2 - 2*t**3),
                            min_width + (max_width-min_width)*(3*t**2 - 2*t**3))
    width = width_factor.astype(int)
    for x in range(Nx):
        y_min = max(y_center - width[x]//2, 0)
        y_max = min(y_center + width[x]//2, Ny)
        nozzle[:y_min,x] = True
        nozzle[y_max:,x] = True
    nozzle = smooth_nozzle_geometry(nozzle)
    nozzle[:,0:5] = False
    nozzle[:,-5:] = False

    # History for convergence plots
    history = {'it': [], 'max_vel': [], 'mass_diff': [], 'residual': []}
    
    rho_init = np.sum(F, axis=2)
    initial_mass = np.sum(rho_init)
    ux_prev = np.zeros((Ny, Nx))
    uy_prev = np.zeros((Ny, Nx))
    
    comprehensive_frames = []
    velocity_frames = []
    
    # Main LBM loop
    start_time = time.time()
    for it in range(1, Nt+1):
        # Inlet BC
        F = apply_inlet_bc(F, 1.0, u_inlet, cxs, cys, weights, inlet_x=0)
        
        # Streaming
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:,:,i] = np.roll(np.roll(F[:,:,i], cx, axis=1), cy, axis=0)
        
        # Outlet BC
        F = apply_outlet_bc(F, outlet_x=-1)
        
        # Bounce-back
        F_wall = F[nozzle, :]
        F[nozzle, :] = F_wall[:, bounce]
        
        # Macroscopic quantities
        rho = np.sum(F, axis=2)
        rho[rho==0] = 1.0
        ux = np.sum(F*cxs.reshape(1,1,NL), axis=2)/rho
        uy = np.sum(F*cys.reshape(1,1,NL), axis=2)/rho
        ux[nozzle]=0.0; uy[nozzle]=0.0
        
        # Collision
        Feq = get_equilibrium(rho, ux, uy, cxs, cys, weights)
        F += -(1.0/tau)*(F-Feq)
        
        # Monitoring metrics every 10 iterations
        if it % 10 == 0:
            vel_mag_sq = ux**2 + uy**2
            max_v = np.sqrt(np.max(vel_mag_sq))
            current_mass = np.sum(rho)
            mass_err = (current_mass - initial_mass)/initial_mass*100
            diff_sq = (ux - ux_prev)**2 + (uy - uy_prev)**2
            res = np.sqrt(np.sum(diff_sq)) / (np.sqrt(np.sum(vel_mag_sq))+1e-12)
            
            history['it'].append(it)
            history['max_vel'].append(max_v)
            history['mass_diff'].append(mass_err)
            history['residual'].append(res)
            
            ux_prev[:] = ux[:]
            uy_prev[:] = uy[:]
        
        # Print progress
        if it % 1000 == 0:
            print(f"Iter {it}/{Nt}: Max Vel={history['max_vel'][-1]:.4f} | Res={history['residual'][-1]:.2e}")
        
        # Save frames
        if record_video and (it % save_frames_interval == 0 or it == 1 or it == Nt):
            flow_fields = calculate_flow_fields(rho, ux, uy)
            comprehensive_frames.append(create_comprehensive_frame(flow_fields, nozzle, grid_name, it))
            velocity_frames.append(create_velocity_frame(flow_fields, nozzle, grid_name, it))
    
    # Final flow fields
    flow_fields = calculate_flow_fields(rho, ux, uy)
    create_final_comprehensive_plot(flow_fields, nozzle, output_folder, grid_name)
    
    # Save validation data including all necessary fields
    np.savez(os.path.join(output_folder, "validation_data_final.npz"),
             rho=rho, ux=ux, uy=uy, velocity_mag=flow_fields['velocity_mag'],
             static_pressure=flow_fields['static_pressure'],
             dynamic_pressure=flow_fields['dynamic_pressure'],
             total_pressure=flow_fields['total_pressure'],
             static_temperature=flow_fields['static_temperature'],
             total_temperature=flow_fields['total_temperature'],
             nozzle=nozzle)
    
    # Create evolution summary
    if record_video and history['it']:
        print(f"\n📹 Creating videos for {grid_name} grid...")
        create_evolution_video(history, output_folder, grid_name)
    
    print(f"✓ Saved validation data & visualizations: {output_folder}")
    
    return output_folder, rho, ux, uy, nozzle, np.max(flow_fields['velocity_mag']), np.sum(rho), flow_fields

def create_final_comprehensive_plot(flow_fields, nozzle, output_folder, grid_name):
    """
    Create final comprehensive plot with 6 flow fields
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'LBM Nozzle Flow - {grid_name} Grid\nFinal State', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    titles = ['Static Pressure', 'Dynamic Pressure', 'Total Pressure',
              'Static Temperature', 'Total Temperature', 'Velocity Magnitude']
    fields = ['static_pressure', 'dynamic_pressure', 'total_pressure',
              'static_temperature', 'total_temperature', 'velocity_mag']
    cmaps = ['viridis', 'plasma', 'coolwarm', 'hot', 'RdYlBu_r', 'jet']
    units = ['Pa', 'Pa', 'Pa', 'K', 'K', 'm/s']
    
    for idx, (title, field, cmap, unit) in enumerate(zip(titles, fields, cmaps, units)):
        ax = axes[idx//3, idx%3]
        im = ax.imshow(flow_fields[field], origin='lower', 
                       cmap=cmap, aspect='auto')
        ax.contour(nozzle, levels=[0.5], colors='white' if cmap in ['plasma', 'hot', 'viridis'] 
                   else 'black', linewidths=1, alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X [lattice units]')
        ax.set_ylabel('Y [lattice units]')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(unit, rotation=270, labelpad=15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(output_folder, f"{grid_name}_final_comprehensive.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

def create_evolution_video(history, output_folder, grid_name):
    """
    Create a video showing the ACTUAL evolution of key parameters over time
    Updated to use real simulation data instead of dummy formulas.
    """
    if not history or len(history['it']) < 2:
        return
    
    iterations = history['it']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Max Velocity Evolution (Real Data)
    axes[0].plot(iterations, history['max_vel'], 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Max Velocity [m/s]')
    axes[0].set_title('Max Velocity History')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Mass Conservation Error (Real Data)
    # Plotting percentage change from initial mass
    axes[1].plot(iterations, history['mass_diff'], 'r-', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Mass Change [%]')
    axes[1].set_title('Mass Conservation Error')
    axes[1].grid(True, alpha=0.3)
    # Use scientific notation if error is very small
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # 3. Residuals / Convergence (Real Data)
    # L2 Norm of velocity difference
    axes[2].semilogy(iterations, history['residual'], 'g-', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Residual (L2 Norm)')
    axes[2].set_title('Convergence (Residuals)')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{grid_name} Grid - Solution Evolution (Real Data)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    evolution_path = os.path.join(output_folder, f"{grid_name}_evolution.png")
    plt.savefig(evolution_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Created evolution summary: {evolution_path}")

# --------------------------
# Verification + GCI + Plotting
# --------------------------
class LBMVerification:
    def __init__(self):
        self.results = {}  # store max velocity & mass for each grid
        self.grid_videos = {}  # store video paths for each grid

    def load_data(self, folder):
        npz_path = os.path.join(folder, "validation_data_final.npz")
        data = np.load(npz_path)
        return data

    def run_verification(self, grid_name, data):
        rho = data['rho']
        ux = data['ux']
        uy = data['uy']
        velocity_mag = data['velocity_mag']
        nozzle = data['nozzle']
        
        # Mass
        total_mass = np.sum(rho)
        # Continuity (Inlet, Throat, Outlet)
        Nx = rho.shape[1]
        inlet_x = 5
        throat_x = Nx // 3
        outlet_x = Nx - 5
        mass_inlet = np.sum(rho[:, inlet_x] * ux[:, inlet_x])
        mass_throat = np.sum(rho[:, throat_x] * ux[:, throat_x])
        mass_outlet = np.sum(rho[:, outlet_x] * ux[:, outlet_x])
        # Reynolds Number at throat
        throat_width = np.sum(~nozzle[:, throat_x])
        u_throat = np.mean(ux[~nozzle[:, throat_x], throat_x])
        nu = (1.0 / 3.0) * (0.6 - 0.5)
        Re = u_throat * throat_width / nu
        # Max velocity
        vel_max = np.max(velocity_mag)
        # Save to results
        self.results[grid_name] = {
            'mass': total_mass,
            'mass_flow': [mass_inlet, mass_throat, mass_outlet],
            'Re': Re,
            'vel_max': vel_max,
            'static_pressure_range': [np.min(data['static_pressure']), np.max(data['static_pressure'])],
            'dynamic_pressure_range': [np.min(data['dynamic_pressure']), np.max(data['dynamic_pressure'])]
        }

    def compute_gci(self, r, p=2.0, Fs=1.25):
        # Using vel_max as QoI
        f_coarse = self.results['Coarse']['vel_max']
        f_medium = self.results['Medium']['vel_max']
        f_fine = self.results['Fine']['vel_max']
        e21 = (f_medium - f_coarse) / f_coarse
        e32 = (f_fine - f_medium) / f_medium
        GCI21 = Fs * abs(e21) / (r**p - 1)
        GCI32 = Fs * abs(e32) / (r**p - 1)
        print(f"GCI Medium-Coarse: {GCI21*100:.2f}% | GCI Fine-Medium: {GCI32*100:.2f}%")
        return GCI21, GCI32

    def create_comparison_video(self, output_folders):
        """
        Create a comparison video showing all three grids side by side
        """
        print("\n🎬 Creating comparison video across all grids...")
        
        # Load comprehensive videos from each grid
        video_paths = {}
        for grid_name in ['Coarse', 'Medium', 'Fine']:
            if grid_name in output_folders:
                video_path = os.path.join(output_folders[grid_name], f"{grid_name}_comprehensive.mp4")
                if os.path.exists(video_path):
                    video_paths[grid_name] = video_path
        
        if len(video_paths) < 2:
            print("   ⚠️ Not enough videos found for comparison")
            return
        
        # Create a comparison montage
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle('Grid Convergence Comparison - Nozzle Flow Simulation', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # We'll create a simple comparison visualization
        grids = list(video_paths.keys())
        colors = ['blue', 'orange', 'green']
        
        # Plot 1: Max velocity comparison
        ax1 = axes[0, 0]
        vel_max = [self.results[g]['vel_max'] for g in grids if g in self.results]
        ax1.bar(grids[:len(vel_max)], vel_max, color=colors[:len(vel_max)])
        ax1.set_ylabel("Max Velocity [m/s]")
        ax1.set_title("Maximum Velocity Comparison")
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Mass flow rate comparison
        ax2 = axes[0, 1]
        mass_flows = [np.mean(self.results[g]['mass_flow']) for g in grids if g in self.results]
        ax2.bar(grids[:len(mass_flows)], mass_flows, color=colors[:len(mass_flows)])
        ax2.set_ylabel("Average Mass Flow Rate")
        ax2.set_title("Mass Flow Rate Comparison")
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: GCI results
        ax3 = axes[1, 0]
        if 'Coarse' in self.results and 'Medium' in self.results and 'Fine' in self.results:
            GCI21, GCI32 = self.compute_gci(r=2.0)
            gci_values = [GCI21*100, GCI32*100]
            gci_labels = ['Coarse→Medium', 'Medium→Fine']
            ax3.bar(gci_labels, gci_values, color=['skyblue', 'lightgreen'])
            ax3.set_ylabel("GCI [%]")
            ax3.set_title("Grid Convergence Index")
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Resolution info
        ax4 = axes[1, 1]
        ax4.axis('off')
        info_text = "Grid Resolutions:\n\n"
        resolutions = {'Coarse': '150×50', 'Medium': '300×100', 'Fine': '600×200'}
        for grid in grids:
            if grid in resolutions:
                info_text += f"{grid}: {resolutions[grid]}\n"
                if grid in self.results:
                    info_text += f"  Max Vel: {self.results[grid]['vel_max']:.4f} m/s\n"
                    info_text += f"  Mass: {self.results[grid]['mass']:.0f}\n\n"
        
        ax4.text(0.1, 0.5, info_text, fontsize=10, va='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        comparison_path = "grid_convergence_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Created comparison plot: {comparison_path}")

# ========================
# Run 3-Grid Simulation + Verification
# ========================
if __name__ == "__main__":
# กำหนดรอบการคำนวณแยกตามความละเอียด
    grids_config = {
        'Coarse': {'size': (150, 50),  'Nt': 6000},
        'Medium': {'size': (300, 100), 'Nt': 10000}, # เพิ่มขึ้นหน่อย
        'Fine':   {'size': (600, 200), 'Nt': 20000}  # เพิ่มเยอะๆ เพื่อให้ Residual ลงมาแตะ 10^-5
    }
    
    verifier = LBMVerification()
    output_folders = {}
    r = 2.0
    
    for name, config in grids_config.items():
        Nx, Ny = config['size']
        Nt_val = config['Nt']
        
        # ส่ง Nt_val เข้าไปในฟังก์ชัน
        folder, rho, ux, uy, nozzle, vel_max, total_mass, flow_fields = run_lbm_nozzle(
            Nx, Ny, name, Nt=Nt_val, save_frames_interval=1000, record_video=True
        )
        
        # Load the saved data for verification
        data = verifier.load_data(folder)
        verifier.run_verification(name, data)
        print(f"Grid {name}: Max Vel={vel_max:.4f} | Total Mass={total_mass:.2f}")
        print(f"   Videos and GIFs saved in: {folder}\n")

    # Compute GCI
    print("\n" + "="*50)
    print("GRID CONVERGENCE ANALYSIS")
    print("="*50)
    verifier.compute_gci(r=r)
    
    # Create comparison video
    verifier.create_comparison_video(output_folders)
    
    print("\n" + "="*50)
    print("SIMULATION COMPLETE!")
    print("="*50)
    print("Each grid has its own folder with:")
    print("  • Comprehensive video (MP4)")
    print("  • Comprehensive GIF")
    print("  • Velocity field video (MP4)")
    print("  • Velocity field GIF")
    print("  • Final comprehensive plot (PNG)")
    print("  • Validation data (NPZ)")
    print("\nFolders created:")
    for grid_name, folder in output_folders.items():
        print(f"  • {grid_name}: {folder}/")
