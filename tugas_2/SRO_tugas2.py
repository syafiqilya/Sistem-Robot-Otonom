# %%
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# %%
# 1. Setup Connection
print("Program Started")
client = RemoteAPIClient()
sim = client.require('sim')
# %%
# 2. Start Simulation
sim.startSimulation()
print("Simulation Started")

# %%
# 3. Simple Test: Post a message to CoppeliaSim status bar
sim.addLog(1, "Hello from Python!")
p3dx_RW = sim.getObject("/PioneerP3DX/rightMotor")
p3dx_LW = sim.getObject("/PioneerP3DX/leftMotor")
p3dx = sim.getObject("/PioneerP3DX")
rw= 0.195/2 #wheel radius
rb= 0.381/2 #body radius
d= 0.05
dt = 0.01
x_int = 0.0
y_int = 0.0
x_abs = 0.0
y_abs = 0.0
theta= sim.getObjectOrientation(p3dx, sim.handle_world)
gamma_int = theta[2]


x_abs_odom = []
y_abs_odom = []
x_int_odom = []
y_int_odom = []

# %%
try:
    # 4. Main Loop (Run for 10 seconds)
    start_time = time.time()
    elapsed_prev = 0.0
    while (time.time() - start_time) < 30:
        
        # --- STUDENT CODE GOES HERE ---
        # Example: Print elapsed time
        elapsed = time.time() - start_time
        print(f"Running... {elapsed:.1f}s", end="\r")

        dt = elapsed - elapsed_prev
        elapsed_prev = elapsed
        #time.sleep(0.1)
        wr_vel=sim.getJointTargetVelocity(p3dx_RW)
        wl_vel=sim.getJointTargetVelocity(p3dx_LW)  

        vx=(wr_vel+wl_vel)*rw/rb
        wx=(wr_vel-wl_vel)*rw/rb  
        sim.addLog(1, f"Vx:{vx:.1f}m/s,Wx:{wx:.1f}rad/s") 
        
        #Get Orientation
        theta= sim.getObjectOrientation(p3dx, sim.handle_world)
        
        #Odometry Space Absolut
        x_abs_dot = vx*math.cos(theta[2])
        y_abs_dot = vx*math.sin(theta[2])
        x_abs = x_abs + x_abs_dot*dt
        y_abs = y_abs + y_abs_dot*dt


        gamma_int = gamma_int + wx*dt
        x_int_dot = vx*math.cos(gamma_int)
        y_int_dot = vx*math.sin(gamma_int)
        x_int = x_int + x_int_dot*dt
        y_int = y_int + y_int_dot*dt

        #safe plot for odometry
        x_abs_odom.append(x_abs)
        y_abs_odom.append(y_abs)
        x_int_odom.append(x_int)
        y_int_odom.append(y_int)
        
        time.sleep(0.1)
         

finally:
    # 5. Stop Simulation safely
    sim.stopSimulation()
    print("\nSimulation Stopped")
    plt.figure(figsize=(10,6))
    plt.plot(x_abs_odom, y_abs_odom, 'b--', label='Absolute Orientation')
    plt.plot(x_int_odom, y_int_odom, 'r--', label='Integral based Orientation')
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')
    plt.title('Odometry comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
