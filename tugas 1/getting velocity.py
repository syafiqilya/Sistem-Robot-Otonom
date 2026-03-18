import time
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# 1. Setup Connection
print("Program Started")
client = RemoteAPIClient()
sim = client.require('sim')

# 2. Start Simulation
sim.startSimulation()
sim.addLog(sim.verbosity_scriptinfos, "Python Script Connected & Simulation Started")

# 3. Handle and Constant Setup
p3dx_RW = sim.getObject("/PioneerP3DX/rightMotor")
p3dx_LW = sim.getObject("/PioneerP3DX/leftMotor")

rw = 0.195 / 2  # wheel radius (r)
L = 0.381       # total distance between wheels (track width)

# Lists for Matplotlib plotting
t_data, phi_r_data, phi_l_data, vx_data, omega_data = [], [], [], [], []

try:
    start_time = time.time()
    while (time.time() - start_time) < 20:
        current_t = time.time() - start_time
        
        # Get actual joint velocities (rad/s)
        wr_vel = sim.getJointVelocity(p3dx_RW)
        wl_vel = sim.getJointVelocity(p3dx_LW)

        # Kinematic Calculations
        vx = (rw / 2.0) * (wr_vel + wl_vel)
        omega = (rw / L) * (wr_vel - wl_vel)

        # --- LOGGING TO COPPELIASIM STATUS BAR ---
        # Formatting the string for readability in the CoppeliaSim Console
        log_msg = (f"[t={current_t:.1f}s] "
                   f"Joints: R={wr_vel:.2f}, L={wl_vel:.2f} rad/s | "
                   f"Body: Vx={vx:.2f} m/s, w={omega:.2f} rad/s")
        
        sim.addLog(sim.verbosity_scriptinfos, log_msg)

        # Store data for final plots
        t_data.append(current_t)
        phi_r_data.append(wr_vel)
        phi_l_data.append(wl_vel)
        vx_data.append(vx)
        omega_data.append(omega)

        time.sleep(0.1) 

finally:
    sim.stopSimulation()
    sim.addLog(sim.verbosity_scriptinfos, "Simulation Stopped. Displaying Plot Window...")
    print("\nSimulation Stopped.")

    # --- Plotting Section ---
    plt.figure(figsize=(10, 6))

    # Subplot 1: Joint Velocities
    plt.subplot(2, 1, 1)
    plt.plot(t_data, phi_r_data, label=r'$\dot{\phi}_R$', color='blue')
    plt.plot(t_data, phi_l_data, label=r'$\dot{\phi}_L$', color='red', linestyle='--')
    plt.title('P3DX Joint Velocities (rad/s) vs Time')
    plt.ylabel('rad/s')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Body Velocities
    plt.subplot(2, 1, 2)
    plt.plot(t_data, vx_data, label='$V_x$ (Linear)', color='green')
    plt.plot(t_data, omega_data, label='$\omega$ (Angular)', color='purple')
    plt.title('P3DX Body Velocity ($V_x$ and $\omega$) vs Time')
    plt.xlabel('Time (sec)')
    plt.ylabel('m/s or rad/s')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()