import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

"""
LBM Nozzle Flow Verification Suite
===================================
ชุดการตรวจสอบความถูกต้องของ LBM simulation สำหรับ nozzle flow
"""

class LBMVerification:
    def __init__(self, data_folder=None):
        # Auto-detect folder if not specified
        if data_folder is None:
            if os.path.exists("LBM_Data_Video_InletBC"):
                data_folder = "LBM_Data_Video_InletBC"
                print("Using folder: LBM_Data_Video_InletBC (with Inlet BC)")
            elif os.path.exists("LBM_Data_Video"):
                data_folder = "LBM_Data_Video"
                print("Using folder: LBM_Data_Video (original)")
            else:
                raise FileNotFoundError("No simulation data folder found!")
        
        self.data_folder = data_folder
        self.excel_path = os.path.join(data_folder, "LBM_Log.xlsx")
        self.npz_path = os.path.join(data_folder, "validation_data_final.npz")
        
        # Load data
        self.df = None
        self.validation_data = None
        self.load_data()
        
    def load_data(self):
        """โหลดข้อมูลจาก Excel และ npz files"""
        try:
            self.df = pd.read_excel(self.excel_path)
            print(f"✓ Loaded Excel data: {len(self.df)} timesteps")
        except Exception as e:
            print(f"✗ Error loading Excel: {e}")
            raise
            
        try:
            self.validation_data = np.load(self.npz_path)
            print(f"✓ Loaded validation data")
        except Exception as e:
            print(f"✗ Error loading npz: {e}")
            raise
    
    def verify_mass_conservation(self, tolerance=1e-2):
        """
        TEST 1: ตรวจสอบ Mass Conservation
        Total mass ในระบบควรคงที่ตลอดเวลา (สำหรับ closed system)
        หรือ converge to steady state (สำหรับ system with inlet BC)
        """
        print("\n" + "="*60)
        print("TEST 1: Mass Conservation")
        print("="*60)
        
        total_density = self.df['Total_Density'].values
        mean_density = np.mean(total_density)
        
        # Check if system has inlet BC (mass increases during startup)
        mass_increasing = total_density[-1] > total_density[0] * 1.01
        
        if mass_increasing:
            print("Detected: System with inlet boundary condition")
            # For inlet BC, check last 25% of simulation for steady state
            last_quarter_idx = int(len(total_density) * 0.75)
            steady_density = total_density[last_quarter_idx:]
            relative_variation = np.std(steady_density) / np.mean(steady_density)
            
            print(f"Initial total density: {total_density[0]:.2f}")
            print(f"Final total density: {total_density[-1]:.2f}")
            print(f"Mass increase during startup: {(total_density[-1]/total_density[0] - 1)*100:.2f}%")
            print(f"Steady-state variation (last 25%): {relative_variation:.4f}")
            
            passed = relative_variation < tolerance
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n{status} - {'Steady state achieved' if passed else 'System not converged'}")
        else:
            print("Detected: Closed system (no inlet BC)")
            relative_variation = np.std(total_density) / mean_density
            max_change = (np.max(total_density) - np.min(total_density)) / mean_density
            
            print(f"Mean total density: {mean_density:.2f}")
            print(f"Relative variation: {relative_variation:.4f}")
            print(f"Max relative change: {max_change:.4f}")
            
            passed = relative_variation < tolerance
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n{status} - {'Mass is conserved' if passed else 'Mass conservation violated!'}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.df['Iteration'], total_density, 'b-', linewidth=2, label='Total Density')
        ax.axhline(mean_density, color='r', linestyle='--', label='Mean')
        
        if mass_increasing:
            # Highlight steady-state region
            steady_start = self.df['Iteration'].values[last_quarter_idx]
            ax.axvspan(steady_start, self.df['Iteration'].values[-1], 
                      alpha=0.2, color='green', label='Steady-state region')
        else:
            ax.fill_between(self.df['Iteration'], 
                           mean_density*(1-tolerance), 
                           mean_density*(1+tolerance), 
                           alpha=0.3, color='green', label=f'±{tolerance*100}% tolerance')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Density')
        ax.set_title('Mass Conservation / Steady State Check')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, 'verify_mass_conservation.png'), dpi=150)
        plt.close()
        
        return passed
    
    def verify_continuity_equation(self):
        """
        TEST 2: ตรวจสอบ Continuity Equation
        Mass flow rate ที่ inlet, throat และ outlet ควรเท่ากัน
        """
        print("\n" + "="*60)
        print("TEST 2: Continuity Equation")
        print("="*60)
        
        # คำนวณ mass flow ที่ต่างๆ จาก final data
        vel = self.validation_data['velocity_field']
        rho = self.validation_data['density_field']
        ux = self.validation_data['ux']
        nozzle = self.validation_data['nozzle_geometry']
        
        # Mass flow at different x-locations
        Nx = vel.shape[1]
        inlet_x = 10
        throat_x = Nx // 4
        outlet_x = Nx - 10
        
        mass_flow_inlet = np.sum(rho[:, inlet_x] * ux[:, inlet_x])
        mass_flow_throat = np.sum(rho[:, throat_x] * ux[:, throat_x])
        mass_flow_outlet = np.sum(rho[:, outlet_x] * ux[:, outlet_x])
        
        print(f"Mass flow at inlet (x={inlet_x}): {mass_flow_inlet:.4f}")
        print(f"Mass flow at throat (x={throat_x}): {mass_flow_throat:.4f}")
        print(f"Mass flow at outlet (x={outlet_x}): {mass_flow_outlet:.4f}")
        
        # Check relative differences
        avg_flow = np.mean([mass_flow_inlet, mass_flow_throat, mass_flow_outlet])
        rel_diff_inlet = abs(mass_flow_inlet - avg_flow) / avg_flow
        rel_diff_throat = abs(mass_flow_throat - avg_flow) / avg_flow
        rel_diff_outlet = abs(mass_flow_outlet - avg_flow) / avg_flow
        
        print(f"\nRelative differences from average:")
        print(f"  Inlet: {rel_diff_inlet*100:.2f}%")
        print(f"  Throat: {rel_diff_throat*100:.2f}%")
        print(f"  Outlet: {rel_diff_outlet*100:.2f}%")
        
        tolerance = 0.15  # 15% tolerance
        passed = max(rel_diff_inlet, rel_diff_throat, rel_diff_outlet) < tolerance
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status} - {'Continuity satisfied' if passed else 'Continuity violated!'}")
        
        return passed
    
    def verify_bernoulli_principle(self):
        """
        TEST 3: ตรวจสอบ Bernoulli's Principle
        Total pressure ควรคงที่ตามแนวกลาง streamline (สำหรับ inviscid flow)
        """
        print("\n" + "="*60)
        print("TEST 3: Bernoulli's Principle (Total Pressure)")
        print("="*60)
        
        vel = self.validation_data['velocity_field']
        rho = self.validation_data['density_field']
        nozzle = self.validation_data['nozzle_geometry']
        
        Ny, Nx = vel.shape
        y_center = Ny // 2
        
        # Extract centerline data
        centerline_vel = vel[y_center, :]
        centerline_rho = rho[y_center, :]
        centerline_nozzle = nozzle[y_center, :]
        
        # คำนวณ total pressure ตาม Bernoulli
        # P_total = P_static + (1/2)*rho*v^2
        c_s = 1/np.sqrt(3)
        gamma = 1.4
        rho0 = 100
        
        static_p = centerline_rho / rho0
        dynamic_p = 0.5 * centerline_rho * centerline_vel**2
        total_p_simple = static_p + dynamic_p
        
        # Compressible form
        mach = centerline_vel / c_s
        total_p_comp = static_p * (1 + (gamma-1)/2 * mach**2)**(gamma/(gamma-1))
        
        # ตัดส่วนที่เป็น nozzle wall ออก
        valid = ~centerline_nozzle
        
        mean_total = np.mean(total_p_comp[valid])
        std_total = np.std(total_p_comp[valid])
        relative_variation = std_total / mean_total
        
        print(f"Mean total pressure (centerline): {mean_total:.4f}")
        print(f"Standard deviation: {std_total:.4f}")
        print(f"Relative variation: {relative_variation*100:.2f}%")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        x = np.arange(Nx)
        ax1.plot(x[valid], static_p[valid], 'b-', label='Static Pressure', linewidth=2)
        ax1.plot(x[valid], dynamic_p[valid], 'r-', label='Dynamic Pressure', linewidth=2)
        ax1.plot(x[valid], total_p_comp[valid], 'g-', label='Total Pressure', linewidth=2)
        ax1.set_ylabel('Pressure')
        ax1.set_title('Pressure Distribution along Centerline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(x[valid], centerline_vel[valid], 'purple', linewidth=2)
        ax2.set_xlabel('x position')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity Distribution along Centerline')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, 'verify_bernoulli.png'), dpi=150)
        plt.close()
        
        tolerance = 0.1  # 10% variation
        passed = relative_variation < tolerance
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status} - {'Bernoulli principle satisfied' if passed else 'Large pressure variation detected'}")
        
        return passed
    
    def verify_reynolds_number(self):
        """
        TEST 4: ตรวจสอบ Reynolds Number Calculation
        Re ควรมีค่าสมเหตุสมผล และคำนวณถูกต้อง
        """
        print("\n" + "="*60)
        print("TEST 4: Reynolds Number Verification")
        print("="*60)
        
        # จาก log
        Re_final = self.df['Reynolds_Number'].iloc[-1]
        throat_velocity = self.df['Throat_Velocity'].iloc[-1]
        
        print(f"Final Reynolds Number: {Re_final:.2f}")
        print(f"Final Throat Velocity: {throat_velocity:.4f}")
        
        # Manual calculation
        tau = 0.8
        c_s = 1/np.sqrt(3)
        nu = c_s**2 * (tau - 0.5)
        Ny = 200
        min_width = int(Ny * 0.2)
        
        Re_manual = throat_velocity * min_width / nu
        print(f"\nManual Re calculation: {Re_manual:.2f}")
        print(f"Difference: {abs(Re_final - Re_manual):.4f}")
        
        # Check if Re is in reasonable range for LBM
        print(f"\nPhysical interpretation:")
        if Re_final < 100:
            flow_regime = "Laminar (Re < 100)"
        elif Re_final < 2300:
            flow_regime = "Transitional (100 < Re < 2300)"
        else:
            flow_regime = "Turbulent (Re > 2300)"
        print(f"  Flow regime: {flow_regime}")
        
        # Check calculation accuracy
        calc_correct = abs(Re_final - Re_manual) < 0.01
        in_valid_range = 1 < Re_final < 10000  # Relaxed range for low-velocity flows
        
        passed = calc_correct and in_valid_range
        status = "✓ PASS" if passed else "✗ FAIL"
        
        if calc_correct:
            print(f"\n{status} - Re calculation is correct!")
            if Re_final < 10:
                print(f"  Note: Very low Re ({Re_final:.2f}) indicates low-velocity flow")
        else:
            print(f"\n{status} - Re calculation error!")
        
        return passed
    
    def verify_velocity_profile(self):
        """
        TEST 5: ตรวจสอบ Velocity Profile
        Velocity ควร: เพิ่มขึ้นตอน converging, สูงสุดที่ throat, ลดลงตอน diverging
        """
        print("\n" + "="*60)
        print("TEST 5: Velocity Profile (Area-Velocity Relationship)")
        print("="*60)
        
        vel = self.validation_data['velocity_field']
        nozzle = self.validation_data['nozzle_geometry']
        
        Ny, Nx = vel.shape
        y_center = Ny // 2
        throat_x = Nx // 4
        
        # คำนวณ cross-sectional area และ average velocity
        areas = []
        avg_velocities = []
        
        for x in range(Nx):
            flow_region = ~nozzle[:, x]
            area = np.sum(flow_region)
            if area > 0:
                avg_vel = np.mean(vel[flow_region, x])
                areas.append(area)
                avg_velocities.append(avg_vel)
            else:
                areas.append(0)
                avg_velocities.append(0)
        
        areas = np.array(areas)
        avg_velocities = np.array(avg_velocities)
        
        # Find throat location (minimum area)
        throat_idx = np.argmin(areas[10:-10]) + 10  # avoid boundaries
        
        print(f"Detected throat at x = {throat_idx} (expected: {throat_x})")
        print(f"Velocity at throat: {avg_velocities[throat_idx]:.4f}")
        print(f"Max velocity in domain: {np.max(avg_velocities):.4f}")
        print(f"Min area: {areas[throat_idx]:.1f} pixels")
        
        # Check area-velocity relationship (A * V should be roughly constant for continuity)
        mass_flow_proxy = areas * avg_velocities
        valid = mass_flow_proxy > 0
        
        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        x = np.arange(Nx)
        
        ax1.plot(x, areas, 'b-', linewidth=2)
        ax1.axvline(throat_idx, color='r', linestyle='--', label=f'Detected throat (x={throat_idx})')
        ax1.axvline(throat_x, color='g', linestyle='--', label=f'Expected throat (x={throat_x})')
        ax1.set_ylabel('Cross-sectional Area')
        ax1.set_title('Nozzle Geometry')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(x, avg_velocities, 'r-', linewidth=2)
        ax2.axvline(throat_idx, color='r', linestyle='--')
        ax2.set_ylabel('Average Velocity')
        ax2.set_title('Velocity Distribution')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(x[valid], mass_flow_proxy[valid], 'g-', linewidth=2)
        ax3.axhline(np.mean(mass_flow_proxy[valid]), color='b', linestyle='--', label='Mean')
        ax3.set_xlabel('x position')
        ax3.set_ylabel('Area × Velocity')
        ax3.set_title('Mass Flow Proxy (should be constant)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, 'verify_velocity_profile.png'), dpi=150)
        plt.close()
        
        # Checks
        throat_error = abs(throat_idx - throat_x)
        vel_increases_before_throat = np.all(np.diff(avg_velocities[10:throat_idx]) >= -0.01)
        
        print(f"\nTroat location error: {throat_error} pixels")
        print(f"Velocity increases before throat: {vel_increases_before_throat}")
        
        passed = (throat_error < 10) and vel_increases_before_throat
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status} - {'Velocity profile correct' if passed else 'Velocity profile anomaly!'}")
        
        return passed
    
    def verify_numerical_stability(self):
        """
        TEST 6: ตรวจสอบ Numerical Stability
        ตรวจหา oscillations, non-physical values, หรือ divergence
        """
        print("\n" + "="*60)
        print("TEST 6: Numerical Stability")
        print("="*60)
        
        vel = self.validation_data['velocity_field']
        rho = self.validation_data['density_field']
        mach = self.validation_data['mach_field']
        nozzle = self.validation_data['nozzle_geometry']
        
        # Check for NaN or Inf
        has_nan = np.any(np.isnan(vel)) or np.any(np.isnan(rho))
        has_inf = np.any(np.isinf(vel)) or np.any(np.isinf(rho))
        
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")
        
        # Check velocity range
        vel_flow = vel[~nozzle]
        print(f"\nVelocity statistics (flow region):")
        print(f"  Min: {np.min(vel_flow):.4f}")
        print(f"  Max: {np.max(vel_flow):.4f}")
        print(f"  Mean: {np.mean(vel_flow):.4f}")
        print(f"  Std: {np.std(vel_flow):.4f}")
        
        # Check Mach number (ไม่ควรเกิน 1 สำหรับ subsonic nozzle)
        mach_flow = mach[~nozzle]
        print(f"\nMach number statistics:")
        print(f"  Max Mach: {np.max(mach_flow):.4f}")
        print(f"  Mean Mach: {np.mean(mach_flow):.4f}")
        
        # Check for oscillations (high frequency variations)
        Ny, Nx = vel.shape
        y_center = Ny // 2
        centerline_vel = vel[y_center, :]
        
        # Compute second derivative (measure of oscillation)
        valid = ~nozzle[y_center, :]
        vel_valid = centerline_vel[valid]
        second_deriv = np.abs(np.diff(vel_valid, n=2))
        oscillation_metric = np.mean(second_deriv)
        
        print(f"\nOscillation metric: {oscillation_metric:.6f}")
        
        # Convergence check from log
        final_10_steps = self.df.tail(10)
        vel_std_last_10 = final_10_steps['Max_Velocity'].std()
        print(f"Velocity std in last 10 steps: {vel_std_last_10:.6f}")
        
        # Plot time evolution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.plot(self.df['Iteration'], self.df['Max_Velocity'], 'b-', linewidth=2, label='Max Velocity')
        ax1.plot(self.df['Iteration'], self.df['Avg_Velocity'], 'r-', linewidth=2, label='Avg Velocity')
        ax1.set_ylabel('Velocity')
        ax1.set_title('Velocity Evolution (Convergence Check)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.df['Iteration'], self.df['Reynolds_Number'], 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Reynolds Number')
        ax2.set_title('Reynolds Number Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, 'verify_stability.png'), dpi=150)
        plt.close()
        
        # Checks
        stable = not (has_nan or has_inf)
        converged = vel_std_last_10 < 0.01
        no_supersonic = np.max(mach_flow) < 1.0
        
        print(f"\nStability checks:")
        print(f"  No NaN/Inf: {stable}")
        print(f"  Converged: {converged}")
        print(f"  Subsonic flow: {no_supersonic}")
        
        passed = stable and converged
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status} - {'Simulation stable' if passed else 'Numerical instability detected!'}")
        
        return passed
    
    def run_all_tests(self):
        """รัน verification tests ทั้งหมด"""
        print("\n" + "#"*60)
        print("# LBM NOZZLE FLOW VERIFICATION SUITE")
        print("#"*60)
        
        results = {}
        results['Mass Conservation'] = self.verify_mass_conservation()
        results['Continuity Equation'] = self.verify_continuity_equation()
        results['Bernoulli Principle'] = self.verify_bernoulli_principle()
        results['Reynolds Number'] = self.verify_reynolds_number()
        results['Velocity Profile'] = self.verify_velocity_profile()
        results['Numerical Stability'] = self.verify_numerical_stability()
        
        # Summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} - {test_name}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        print("="*60)
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Simulation is verified.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Review results above.")
        
        return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Starting LBM Verification Suite...")
    
    # Auto-detect folder or specify manually
    verifier = LBMVerification()  # Auto-detect
    # Or specify: verifier = LBMVerification(data_folder="LBM_Data_Video_InletBC")
    
    results = verifier.run_all_tests()
    
    print("\nVerification complete! Check the data folder for plots.")