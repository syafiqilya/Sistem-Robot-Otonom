import time
import math
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================================
# USER CONFIGURATION
# ==========================================================

# ----- Simulation -----
SIM_DURATION_SEC = 990.0
CONTROL_DT_SEC = 0.01          # 0.02 s = 50 Hz control loop

# ----- Object names in CoppeliaSim scene -----
ROBOT_PATH = "/PioneerP3DX"
RIGHT_MOTOR_PATH = "/PioneerP3DX/rightMotor"
LEFT_MOTOR_PATH = "/PioneerP3DX/leftMotor"

LH_MARKER_PATH = "/LH"         # optional visual marker
PERP_MARKER_PATH = "/Perp"     # optional visual marker

PATH_POINT_TEMPLATE = "/p[{}]" # expects /p[0], /p[1], ...
MAX_SCENE_PATH_POINTS = 345    # tries to load /p[0] ... /p[119]
REQUIRE_ALL_PATH_POINTS = False
PATH_IS_STATIC = True          # True = read path once; False = read every loop

# ----- Internal path processing -----
# If your scene still has only 60 path points, set this True.
# The code will interpolate them into 120 internal tracking points.
RESAMPLE_PATH = False
INTERNAL_PATH_POINT_COUNT = 120

# ----- Robot geometry -----
WHEEL_RADIUS_M = 0.195 / 2.0
HALF_WHEEL_BASE_M = 0.318 / 2.0

# ----- Look-ahead point position in robot local frame -----
LH_OFFSET_X_M = 0.5
LH_OFFSET_Y_M = 0.0
LH_OFFSET_Z_M = 0.0

# ----- Real path target look-ahead distance -----
PATH_LOOKAHEAD_DISTANCE_M = 2.0

# ----- PID steering controller -----
PID_KP = 5.0
PID_KI = 0.5
PID_KD = 0.25

MAX_OMEGA_RAD_S = 0.75
INTEGRAL_LIMIT = 0.3

# ----- Linear speed control -----
BASE_LINEAR_SPEED_M_S = 0.20
MIN_LINEAR_SPEED_M_S = 0.03
ENABLE_TURN_SPEED_REDUCTION = True
FULL_SLOWDOWN_HEADING_ERROR_RAD = math.radians(60.0)

# ----- Wheel speed safety limit -----
MAX_WHEEL_SPEED_RAD_S = 5.0

# ----- Path progress logic -----
ENABLE_PROGRESS_SEARCH_WINDOW = True
SEARCH_BACK_SEGMENTS = 3
SEARCH_FORWARD_SEGMENTS = 25

STOP_AT_FINAL_POINT = True
GOAL_TOLERANCE_M = 0.15

# ==========================================================
# ULTRASONIC MAPPING CONFIGURATION
# ==========================================================

ENABLE_ULTRASONIC_MAPPING = True

# You said you activated ultrasonicSensor[n] for n = 0, 3, 4, 7.
ULTRASONIC_SENSOR_INDICES = [0, 3, 4, 7]
ULTRASONIC_SENSOR_TEMPLATE = "/PioneerP3DX/ultrasonicSensor[{}]"

# If your sensors are set to explicit handling in CoppeliaSim, change this to True.
# For normal P3DX sensors, False is usually correct.
CALL_HANDLE_PROXIMITY_SENSOR = False

# Occupancy grid size.
# Increase/decrease these limits based on your environment size.
MAP_RESOLUTION_M = 0.05       # 5 cm per cell
MAP_X_MIN = -8.0
MAP_X_MAX = 8.0
MAP_Y_MIN = -8.0
MAP_Y_MAX = 8.0

# Ultrasonic valid range.
MIN_VALID_SENSOR_RANGE_M = 0.02
MAX_VALID_SENSOR_RANGE_M = 3.0

# Log-odds mapping update values.
# Larger occupied value makes obstacle cells darker faster.
# Larger negative free value makes free-space cells whiter faster.
LOG_ODDS_OCCUPIED = 0.85
LOG_ODDS_FREE = -0.30
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0

# Mapping update rate.
# 1 = update every control loop.
# 5 = update every 5 loops, useful for long simulations.
MAPPING_UPDATE_EVERY_N_LOOPS = 2

# If True, no-detection readings also clear free space up to MAX_VALID_SENSOR_RANGE_M.
# I keep this False because wrong sensor direction assumptions can create wrong free rays.
MARK_FREE_SPACE_WHEN_NO_DETECTION = False

# Used only if CoppeliaSim returns distance but not detectedPoint, or if
# MARK_FREE_SPACE_WHEN_NO_DETECTION=True.
# For many CoppeliaSim proximity sensors, local +Z is the detection direction.
SENSOR_FORWARD_AXIS = "z"      # choose "x", "y", or "z"

# Matplotlib output.
SHOW_MAPPING_RESULT_AT_END = True


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def local_point_to_world_2d(robot_pos, robot_yaw, x_local, y_local, z_local=0.0):
    """
    Converts a local robot-frame point into world coordinates.
    Assumes robot moves on a flat X-Y plane.
    This avoids Euler rotation-order ambiguity.
    """
    c = math.cos(robot_yaw)
    s = math.sin(robot_yaw)

    x_world = robot_pos[0] + c * x_local - s * y_local
    y_world = robot_pos[1] + s * x_local + c * y_local
    z_world = robot_pos[2] + z_local

    return np.array([x_world, y_world, z_world], dtype=float)


def get_optional_object(sim, object_path):
    try:
        return sim.getObject(object_path)
    except Exception:
        return None


def load_path_handles(sim, template, max_points, require_all=False):
    handles = []

    for i in range(max_points):
        object_path = template.format(i)
        try:
            handles.append(sim.getObject(object_path))
        except Exception:
            if require_all:
                raise RuntimeError(f"Missing required path point: {object_path}")
            break

    if len(handles) < 2:
        raise RuntimeError(
            f"Need at least 2 path points, but only found {len(handles)}. "
            f"Check names like {template.format(0)}, {template.format(1)}, ..."
        )

    return handles


def read_path_points(sim, path_handles):
    points = []
    for h in path_handles:
        points.append(sim.getObjectPosition(h, sim.handle_world))
    return np.array(points, dtype=float)


def resample_polyline(points, target_count):
    """
    Creates target_count equally spaced points along the path length.
    Useful when the scene has 60 path markers but the controller should use 120.
    """
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        raise ValueError("At least two points are required to resample a path.")

    if target_count <= 2:
        return np.array([points[0], points[-1]], dtype=float)

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors[:, 0:2], axis=1)

    valid = segment_lengths > 1e-9
    if not np.any(valid):
        raise ValueError("Path length is zero; all path points are identical.")

    # Remove zero-length segments for stable interpolation.
    cleaned = [points[0]]
    for i, is_valid in enumerate(valid):
        if is_valid:
            cleaned.append(points[i + 1])
    points = np.array(cleaned, dtype=float)

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors[:, 0:2], axis=1)

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]

    target_distances = np.linspace(0.0, total_length, target_count)
    resampled = []

    for d in target_distances:
        seg_idx = np.searchsorted(cumulative, d, side="right") - 1
        seg_idx = clamp(seg_idx, 0, len(segment_lengths) - 1)

        seg_len = segment_lengths[seg_idx]
        t = 0.0 if seg_len < 1e-9 else (d - cumulative[seg_idx]) / seg_len
        p = points[seg_idx] + t * (points[seg_idx + 1] - points[seg_idx])
        resampled.append(p)

    return np.array(resampled, dtype=float)


def closest_projection_on_path(point, path_points, start_segment=0, end_segment=None):
    """
    Finds the closest projection of point onto path polyline.
    Returns:
        best_projection, best_segment_index, best_t, best_distance
    where best_t is normalized from 0 to 1 on the selected segment.
    """
    point = np.asarray(point, dtype=float)
    path_points = np.asarray(path_points, dtype=float)

    n_segments = len(path_points) - 1
    if n_segments < 1:
        raise ValueError("Path must contain at least two points.")

    if end_segment is None:
        end_segment = n_segments - 1

    start_segment = int(clamp(start_segment, 0, n_segments - 1))
    end_segment = int(clamp(end_segment, start_segment, n_segments - 1))

    best_distance = float("inf")
    best_projection = path_points[start_segment].copy()
    best_segment = start_segment
    best_t = 0.0

    for i in range(start_segment, end_segment + 1):
        a = path_points[i]
        b = path_points[i + 1]
        ab = b - a

        denom = float(np.dot(ab[0:2], ab[0:2]))
        if denom < 1e-12:
            continue

        t = float(np.dot((point - a)[0:2], ab[0:2]) / denom)
        t = clamp(t, 0.0, 1.0)

        proj = a + t * ab
        dist = float(np.linalg.norm((point - proj)[0:2]))

        if dist < best_distance:
            best_distance = dist
            best_projection = proj
            best_segment = i
            best_t = t

    return best_projection, best_segment, best_t, best_distance


def scale_wheel_speeds(left_speed, right_speed, max_abs_speed):
    """
    Scales both wheel speeds together if either exceeds the maximum.
    This preserves the commanded curvature better than clipping independently.
    """
    max_current = max(abs(left_speed), abs(right_speed))
    if max_current > max_abs_speed:
        scale = max_abs_speed / max_current
        left_speed *= scale
        right_speed *= scale
    return left_speed, right_speed


class PIDController:
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        if dt <= 1e-9:
            derivative = 0.0
        elif self.prev_error is None:
            derivative = 0.0
        else:
            derivative = wrap_to_pi(error - self.prev_error) / dt

        self.integral += error * max(dt, 0.0)

        if self.integral_limit is not None:
            self.integral = clamp(
                self.integral,
                -self.integral_limit,
                self.integral_limit
            )

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        if self.output_limit is not None:
            output = clamp(output, -self.output_limit, self.output_limit)

        self.prev_error = error
        return output


# ==========================================================
# ULTRASONIC MAPPING HELPERS
# ==========================================================

def transform_local_to_world_from_matrix(matrix_3x4, local_point):
    """
    Transform a local sensor-frame point into world coordinates.
    CoppeliaSim getObjectMatrix() returns a 3x4 matrix as 12 values.
    """
    m = matrix_3x4
    x = float(local_point[0])
    y = float(local_point[1])
    z = float(local_point[2])

    return np.array([
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    ], dtype=float)


def distance_to_local_point(distance, axis):
    """
    Convert scalar distance into a local sensor-frame point.
    This is only used as fallback if detectedPoint is unavailable.
    """
    axis = axis.lower()

    if axis == "x":
        return np.array([distance, 0.0, 0.0], dtype=float)
    if axis == "y":
        return np.array([0.0, distance, 0.0], dtype=float)

    return np.array([0.0, 0.0, distance], dtype=float)


def parse_proximity_sensor_return(raw):
    """
    Robust parser for sim.readProximitySensor() or sim.handleProximitySensor().

    Common CoppeliaSim format:
        detected, distance, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector

    Return:
        detected: bool
        distance: float or None
        local_point: np.array([x, y, z]) or None
    """
    if raw is None:
        return False, None, None

    if not isinstance(raw, (list, tuple)):
        return bool(raw), None, None

    if len(raw) == 0:
        return False, None, None

    detected = bool(raw[0])
    if not detected:
        return False, None, None

    distance = None
    local_point = None

    for item in raw[1:]:
        if isinstance(item, (int, float)) and distance is None:
            distance = float(item)

        elif isinstance(item, (list, tuple)) and len(item) >= 3 and local_point is None:
            local_point = np.array(
                [float(item[0]), float(item[1]), float(item[2])],
                dtype=float
            )

    if distance is None and local_point is not None:
        distance = float(np.linalg.norm(local_point))

    return True, distance, local_point


def load_ultrasonic_sensor_handles(sim, sensor_indices):
    """
    Load only the ultrasonic sensors selected by ULTRASONIC_SENSOR_INDICES.
    """
    handles = []

    for idx in sensor_indices:
        sensor_path = ULTRASONIC_SENSOR_TEMPLATE.format(idx)

        try:
            h = sim.getObject(sensor_path)
            handles.append((idx, h))
            print(f"Loaded ultrasonic sensor: {sensor_path}")

        except Exception:
            print(f"WARNING: could not find ultrasonic sensor: {sensor_path}")

    if ENABLE_ULTRASONIC_MAPPING and len(handles) == 0:
        raise RuntimeError(
            "ENABLE_ULTRASONIC_MAPPING=True, but no ultrasonic sensor handles were found."
        )

    return handles


class OccupancyGridMap:
    """
    Simple 2D log-odds occupancy grid.
    - Occupied cells are increased by LOG_ODDS_OCCUPIED.
    - Free cells along sensor rays are decreased by LOG_ODDS_FREE.
    """
    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.resolution = float(resolution)

        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution))
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution))

        # log_odds = 0 means unknown, probability = 0.5
        self.log_odds = np.zeros((self.height, self.width), dtype=float)

    def world_to_grid(self, x, y):
        gx = int(math.floor((x - self.x_min) / self.resolution))
        gy = int(math.floor((y - self.y_min) / self.resolution))

        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return None

        return gx, gy

    def update_cell(self, gx, gy, delta):
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return

        self.log_odds[gy, gx] = clamp(
            self.log_odds[gy, gx] + delta,
            LOG_ODDS_MIN,
            LOG_ODDS_MAX
        )

    def update_ray(self, start_world, end_world, hit):
        """
        Mark cells between sensor and obstacle as free.
        If hit=True, mark endpoint as occupied.
        """
        start_world = np.asarray(start_world, dtype=float)
        end_world = np.asarray(end_world, dtype=float)

        sx = float(start_world[0])
        sy = float(start_world[1])
        ex = float(end_world[0])
        ey = float(end_world[1])

        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)

        if length < 1e-9:
            return

        # Sample along the ray at half-cell spacing.
        n_steps = max(2, int(math.ceil(length / (self.resolution * 0.5))))

        # Free cells, excluding final endpoint.
        for k in range(n_steps):
            t = k / n_steps
            x = sx + t * dx
            y = sy + t * dy

            cell = self.world_to_grid(x, y)
            if cell is not None:
                self.update_cell(cell[0], cell[1], LOG_ODDS_FREE)

        # Occupied endpoint.
        if hit:
            end_cell = self.world_to_grid(ex, ey)
            if end_cell is not None:
                self.update_cell(end_cell[0], end_cell[1], LOG_ODDS_OCCUPIED)

    def probability(self):
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def plot(self, robot_trace=None, path_points=None, hit_points=None):
        prob = self.probability()

        plt.figure(figsize=(10, 8))
        plt.imshow(
            prob,
            origin="lower",
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            vmin=0.0,
            vmax=1.0,
            cmap="gray_r"
        )
        plt.colorbar(label="Occupancy probability")

        if path_points is not None and len(path_points) > 0:
            path_arr = np.asarray(path_points, dtype=float)
            plt.plot(
                path_arr[:, 0],
                path_arr[:, 1],
                linestyle="--",
                linewidth=1.0,
                label="Reference path"
            )

        if robot_trace is not None and len(robot_trace) > 0:
            trace_arr = np.asarray(robot_trace, dtype=float)
            plt.plot(
                trace_arr[:, 0],
                trace_arr[:, 1],
                linewidth=1.5,
                label="Robot trajectory"
            )

        if hit_points is not None and len(hit_points) > 0:
            hits_arr = np.asarray(hit_points, dtype=float)
            plt.scatter(
                hits_arr[:, 0],
                hits_arr[:, 1],
                s=4,
                label="Ultrasonic hit points"
            )

        plt.title("P3DX Ultrasonic Mapping Result")
        plt.xlabel("World X [m]")
        plt.ylabel("World Y [m]")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def update_map_from_ultrasonic_sensors(sim, sensor_handles, occ_map, hit_points_log):
    """
    Read selected ultrasonic sensors and update occupancy grid.
    """
    for sensor_index, sensor_handle in sensor_handles:
        if CALL_HANDLE_PROXIMITY_SENSOR:
            raw = sim.handleProximitySensor(sensor_handle)
        else:
            raw = sim.readProximitySensor(sensor_handle)

        detected, distance, local_point = parse_proximity_sensor_return(raw)

        sensor_matrix = sim.getObjectMatrix(sensor_handle, sim.handle_world)

        sensor_world_pos = np.array([
            sensor_matrix[3],
            sensor_matrix[7],
            sensor_matrix[11]
        ], dtype=float)

        if detected:
            if distance is None:
                continue

            if distance < MIN_VALID_SENSOR_RANGE_M:
                continue

            if distance > MAX_VALID_SENSOR_RANGE_M:
                continue

            if local_point is None:
                local_point = distance_to_local_point(distance, SENSOR_FORWARD_AXIS)

            hit_world = transform_local_to_world_from_matrix(sensor_matrix, local_point)

            occ_map.update_ray(sensor_world_pos, hit_world, hit=True)
            hit_points_log.append(hit_world.copy())

        elif MARK_FREE_SPACE_WHEN_NO_DETECTION:
            local_end = distance_to_local_point(MAX_VALID_SENSOR_RANGE_M, SENSOR_FORWARD_AXIS)
            end_world = transform_local_to_world_from_matrix(sensor_matrix, local_end)
            occ_map.update_ray(sensor_world_pos, end_world, hit=False)



# ==========================================================
# MAIN PROGRAM
# ==========================================================

client = RemoteAPIClient()
sim = client.require("sim")

# Get handles before starting, so scene errors are caught early.
p3dx = sim.getObject(ROBOT_PATH)
p3dx_rw = sim.getObject(RIGHT_MOTOR_PATH)
p3dx_lw = sim.getObject(LEFT_MOTOR_PATH)

lh_handle = get_optional_object(sim, LH_MARKER_PATH)
perp_handle = get_optional_object(sim, PERP_MARKER_PATH)

path_handles = load_path_handles(
    sim,
    PATH_POINT_TEMPLATE,
    MAX_SCENE_PATH_POINTS,
    require_all=REQUIRE_ALL_PATH_POINTS
)

raw_path_points = read_path_points(sim, path_handles)

if RESAMPLE_PATH:
    path_points = resample_polyline(raw_path_points, INTERNAL_PATH_POINT_COUNT)
else:
    path_points = raw_path_points

print(f"Loaded {len(raw_path_points)} scene path points.")
print(f"Using {len(path_points)} internal path points.")

# ----- Ultrasonic mapping initialization -----
if ENABLE_ULTRASONIC_MAPPING:
    ultrasonic_sensor_handles = load_ultrasonic_sensor_handles(
        sim,
        ULTRASONIC_SENSOR_INDICES
    )
else:
    ultrasonic_sensor_handles = []

occupancy_map = OccupancyGridMap(
    MAP_X_MIN,
    MAP_X_MAX,
    MAP_Y_MIN,
    MAP_Y_MAX,
    MAP_RESOLUTION_M
)

robot_trace = []
ultrasonic_hit_points = []
loop_counter = 0

pid = PIDController(
    kp=PID_KP,
    ki=PID_KI,
    kd=PID_KD,
    output_limit=MAX_OMEGA_RAD_S,
    integral_limit=INTEGRAL_LIMIT
)

current_segment_idx = 0

sim.startSimulation()
time.sleep(0.2)

start_wall_time = time.time()
last_loop_time = time.time()

try:
    while (time.time() - start_wall_time) < SIM_DURATION_SEC:
        now = time.time()
        dt = now - last_loop_time
        last_loop_time = now

        if dt <= 1e-6:
            dt = CONTROL_DT_SEC

        # Update path if path markers move during simulation.
        if not PATH_IS_STATIC:
            raw_path_points = read_path_points(sim, path_handles)
            if RESAMPLE_PATH:
                path_points = resample_polyline(
                    raw_path_points,
                    INTERNAL_PATH_POINT_COUNT
                )
            else:
                path_points = raw_path_points

        robot_pos = np.array(
            sim.getObjectPosition(p3dx, sim.handle_world),
            dtype=float
        )
        robot_ori = sim.getObjectOrientation(p3dx, sim.handle_world)
        yaw = float(robot_ori[2])

        # ----- Ultrasonic mapping update -----
        robot_trace.append(robot_pos.copy())

        if ENABLE_ULTRASONIC_MAPPING:
            if loop_counter % MAPPING_UPDATE_EVERY_N_LOOPS == 0:
                update_map_from_ultrasonic_sensors(
                    sim,
                    ultrasonic_sensor_handles,
                    occupancy_map,
                    ultrasonic_hit_points
                )

        # Look-ahead marker in world frame.
        lh_world = local_point_to_world_2d(
            robot_pos,
            yaw,
            LH_OFFSET_X_M,
            LH_OFFSET_Y_M,
            LH_OFFSET_Z_M
        )

        # Search only near current progress to avoid jumping backward.
        if ENABLE_PROGRESS_SEARCH_WINDOW:
            start_seg = max(0, current_segment_idx - SEARCH_BACK_SEGMENTS)
            end_seg = min(
                len(path_points) - 2,
                current_segment_idx + SEARCH_FORWARD_SEGMENTS
            )
        else:
            start_seg = 0
            end_seg = len(path_points) - 2

        best_proj, best_seg, best_t, dist_to_lh = closest_projection_on_path(
            lh_world,
            path_points,
            start_segment=start_seg,
            end_segment=end_seg
        )

        # Monotonic path progress update.
        if best_seg > current_segment_idx:
            current_segment_idx = best_seg
        elif best_seg == current_segment_idx and best_t > 0.98:
            current_segment_idx = min(
                current_segment_idx + 1,
                len(path_points) - 2
            )

        # Update visual markers.
        if perp_handle is not None:
            sim.setObjectPosition(
                perp_handle,
                sim.handle_world,
                best_proj.tolist()
            )

        if lh_handle is not None:
            sim.setObjectPosition(
                lh_handle,
                sim.handle_world,
                lh_world.tolist()
            )

        # Stop condition at final path point.
        final_point = path_points[-1]
        distance_to_goal = float(np.linalg.norm((robot_pos - final_point)[0:2]))

        if STOP_AT_FINAL_POINT and current_segment_idx >= len(path_points) - 2:
            if distance_to_goal <= GOAL_TOLERANCE_M:
                break

        # Heading from robot center to selected target point.
        dx = best_proj[0] - robot_pos[0]
        dy = best_proj[1] - robot_pos[1]

        # If target is extremely close, avoid noisy atan2.
        if math.hypot(dx, dy) < 1e-6:
            heading_error = 0.0
        else:
            target_heading = math.atan2(dy, dx)
            heading_error = wrap_to_pi(target_heading - yaw)

        # PID controls angular velocity.
        omega_cmd = pid.update(heading_error, dt)

        # Reduce forward speed when heading error is large.
        if ENABLE_TURN_SPEED_REDUCTION:
            ratio = 1.0 - abs(heading_error) / FULL_SLOWDOWN_HEADING_ERROR_RAD
            ratio = clamp(ratio, 0.0, 1.0)
            v_cmd = MIN_LINEAR_SPEED_M_S + ratio * (
                BASE_LINEAR_SPEED_M_S - MIN_LINEAR_SPEED_M_S
            )
        else:
            v_cmd = BASE_LINEAR_SPEED_M_S

        # Differential drive inverse kinematics.
        v_left = (
            v_cmd - omega_cmd * HALF_WHEEL_BASE_M
        ) / WHEEL_RADIUS_M

        v_right = (
            v_cmd + omega_cmd * HALF_WHEEL_BASE_M
        ) / WHEEL_RADIUS_M

        v_left, v_right = scale_wheel_speeds(
            v_left,
            v_right,
            MAX_WHEEL_SPEED_RAD_S
        )

        sim.setJointTargetVelocity(p3dx_lw, float(v_left))
        sim.setJointTargetVelocity(p3dx_rw, float(v_right))

        loop_counter += 1
        time.sleep(CONTROL_DT_SEC)

finally:
    sim.setJointTargetVelocity(p3dx_lw, 0.0)
    sim.setJointTargetVelocity(p3dx_rw, 0.0)
    time.sleep(0.1)
    sim.stopSimulation()

    if ENABLE_ULTRASONIC_MAPPING and SHOW_MAPPING_RESULT_AT_END:
        occupancy_map.plot(
            robot_trace=robot_trace,
            path_points=path_points,
            hit_points=ultrasonic_hit_points
        )