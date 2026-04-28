import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

sim.startSimulation()

rw = 0.195 / 2
wheel_dist = 0.381

try:
    p_pemain = sim.getObject("/Robot_Pemain")
    p_pemain_RW = sim.getObject("/Robot_Pemain/rightMotor")
    p_pemain_LW = sim.getObject("/Robot_Pemain/leftMotor")

    p_lawan1 = sim.getObject("/Robot_Lawan_01")
    p_lawan1_RW = sim.getObject("/Robot_Lawan_01/rightMotor")
    p_lawan1_LW = sim.getObject("/Robot_Lawan_01/leftMotor")

    bola_merah = sim.getObject("/Bola_Merah")
except Exception as e:
    sim.stopSimulation()
    exit()

# --- TUNABLE VARIABLES ---
# Striker (Robot_Pemain) tuning
pemain_max_speed = 0.35  # Adjust this to change forward speed (previously 0.6)
pemain_turn_gain = 2.0   # Adjust this to change how sharply the robot turns

# Goalkeeper (Robot_Lawan_01) tuning
patrol_amplitude = 0.8
moving_right = True

try:
    while True:
        pemain_pos = sim.getObjectPosition(p_pemain, sim.handle_world)
        pemain_ori = sim.getObjectOrientation(p_pemain, sim.handle_world)
        
        lawan1_pos = sim.getObjectPosition(p_lawan1, sim.handle_world)
        lawan1_ori = sim.getObjectOrientation(p_lawan1, sim.handle_world)
        
        bola_pos = sim.getObjectPosition(bola_merah, sim.handle_world)

        # Exact center of the goal based on provided coordinates
        gawang_x = 5.125
        gawang_y = -0.075

        # --- 1. ROBOT PEMAIN (STRIKER) LOGIC ---
        # Target 1.0m deep inside the goal to ensure the pass crosses the line
        target_g_x = gawang_x + 1.0
        target_g_y = gawang_y

        bg_vec = np.array([target_g_x - bola_pos[0], target_g_y - bola_pos[1]])
        bg_dist = np.linalg.norm(bg_vec)
        bg_dir = bg_vec / bg_dist if bg_dist > 0 else np.array([1.0, 0.0])

        dist_pemain_bola = math.hypot(pemain_pos[0] - bola_pos[0], pemain_pos[1] - bola_pos[1])

        if dist_pemain_bola > 0.5:
            target_p_x = bola_pos[0] - (bg_dir[0] * 0.4)
            target_p_y = bola_pos[1] - (bg_dir[1] * 0.4)
        else:
            target_p_x = target_g_x
            target_p_y = target_g_y

        dx_p = target_p_x - pemain_pos[0]
        dy_p = target_p_y - pemain_pos[1]
        angle_p = math.atan2(dy_p, dx_p) - pemain_ori[2]
        angle_p = (angle_p + math.pi) % (2 * math.pi) - math.pi

        # Apply tunable variables here
        wx_p = pemain_turn_gain * angle_p
        vx_p = 0.0 if abs(angle_p) > math.pi / 4 else pemain_max_speed

        wr_vel_p = (vx_p + (wheel_dist / 2) * wx_p) / rw
        wl_vel_p = (vx_p - (wheel_dist / 2) * wx_p) / rw
        
        sim.setJointTargetVelocity(p_pemain_RW, wr_vel_p)
        sim.setJointTargetVelocity(p_pemain_LW, wl_vel_p)

        # --- 2. ROBOT LAWAN 01 (GOALKEEPER) LOGIC ---
        target_offset = patrol_amplitude if moving_right else -patrol_amplitude
        # Patrol 0.3m in front of the goal line to block shots
        target_l_x = gawang_x - 0.3 
        target_l_y = gawang_y + target_offset

        dist_to_target = math.hypot(target_l_x - lawan1_pos[0], target_l_y - lawan1_pos[1])
        if dist_to_target < 0.2:
            moving_right = not moving_right

        dx_l = target_l_x - lawan1_pos[0]
        dy_l = target_l_y - lawan1_pos[1]
        angle_l = math.atan2(dy_l, dx_l) - lawan1_ori[2]
        angle_l = (angle_l + math.pi) % (2 * math.pi) - math.pi

        wx_l = 2.0 * angle_l
        vx_l = 0.0 if abs(angle_l) > math.pi / 4 else 0.3

        wr_vel_l = (vx_l + (wheel_dist / 2) * wx_l) / rw
        wl_vel_l = (vx_l - (wheel_dist / 2) * wx_l) / rw

        sim.setJointTargetVelocity(p_lawan1_RW, wr_vel_l)
        sim.setJointTargetVelocity(p_lawan1_LW, wl_vel_l)

        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    sim.setJointTargetVelocity(p_pemain_RW, 0.0)
    sim.setJointTargetVelocity(p_pemain_LW, 0.0)
    sim.setJointTargetVelocity(p_lawan1_RW, 0.0)
    sim.setJointTargetVelocity(p_lawan1_LW, 0.0)
    sim.stopSimulation()