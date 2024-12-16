import cv2
import numpy as np
import math

# Map dimensions
MAP_HEIGHT, MAP_WIDTH = 500, 500

# Read and preprocess moon surface image
moon_surface = cv2.imread("moon_surface.jpg", 0)
moon_surface = cv2.resize(moon_surface, (MAP_WIDTH, MAP_HEIGHT))
moon_surface = cv2.normalize(moon_surface, None, 0, 200, cv2.NORM_MINMAX)  # Reduced brightness

def detect_craters(image_path, min_distance=50):
    img = cv2.imread(image_path, 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_distance,
        param1=40,
        param2=25,
        minRadius=15,
        maxRadius=60
    )
    
    craters = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        sorted_circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
        
        for circle in sorted_circles:
            x, y, r = circle
            if all(np.sqrt((x - cx)**2 + (y - cy)**2) >= (r + cr + min_distance) 
                  for cx, cy, cr in craters):
                craters.append((int(x), int(y), int(r)))
            if len(craters) == 5:
                break
    
    return craters

# Initialize map and obstacles
map = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8) * 255
obstacles = detect_craters("moon_surface.jpg")
for obstacle in obstacles:
    cv2.circle(map, (obstacle[0], obstacle[1]), obstacle[2], 0, -1)

# Robot parameters
rx, ry, rtheta = 50, 450, math.pi/4
STEP = 3
TURN = math.radians(5)
ROBOT_WIDTH, ROBOT_HEIGHT = 30, 20

# Particle filter parameters
NUM_PARTICLES = 1000
particles = None
weights = None

def init_particles():
    global particles, weights
    particles = np.zeros((NUM_PARTICLES, 3))
    particles[:, 0] = np.random.uniform(0, MAP_WIDTH, NUM_PARTICLES)
    particles[:, 1] = np.random.uniform(0, MAP_HEIGHT, NUM_PARTICLES)
    particles[:, 2] = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

def motion_update(particles, forward, turn):
    noise_fwd = np.random.normal(0, 1, NUM_PARTICLES)
    noise_turn = np.random.normal(0, 0.05, NUM_PARTICLES)
    particles[:, 2] += turn + noise_turn
    particles[:, 0] += (forward + noise_fwd) * np.cos(particles[:, 2])
    particles[:, 1] += (forward + noise_fwd) * np.sin(particles[:, 2])

def weight_update(particles, rx, ry):
    global weights
    distances = np.sqrt((particles[:, 0] - rx)**2 + (particles[:, 1] - ry)**2)
    weights = np.exp(-distances / 30)
    weights += 1e-300
    weights /= np.sum(weights)

def resample(particles, weights):
    indices = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES, p=weights)
    return particles[indices]

def predicted_pose(particles, weights):
    mean_x = np.average(particles[:, 0], weights=weights)
    mean_y = np.average(particles[:, 1], weights=weights)
    mean_theta = np.arctan2(
        np.average(np.sin(particles[:, 2]), weights=weights),
        np.average(np.cos(particles[:, 2]), weights=weights)
    )
    return mean_x, mean_y, mean_theta

def draw_map(rx, ry, rtheta, particles, waypoints, trajectory):
    # Create background with moon surface
    lmap = cv2.cvtColor(moon_surface, cv2.COLOR_GRAY2BGR)
    
    # Draw obstacles
    for obstacle in obstacles:
        cv2.circle(lmap, (obstacle[0], obstacle[1]), obstacle[2], (0, 0, 0), -1)
    
    # Draw particles
    for px, py, _ in particles:
        cv2.circle(lmap, (int(px), int(py)), 1, (0, 0, 255), -1)
    
    # Draw robot
    robot_corners = np.array([
        [-ROBOT_WIDTH // 2, -ROBOT_HEIGHT // 2],
        [ROBOT_WIDTH // 2, -ROBOT_HEIGHT // 2],
        [ROBOT_WIDTH // 2, ROBOT_HEIGHT // 2],
        [-ROBOT_WIDTH // 2, ROBOT_HEIGHT // 2]
    ])
    rotation_matrix = np.array([
        [math.cos(rtheta), -math.sin(rtheta)],
        [math.sin(rtheta), math.cos(rtheta)]
    ])
    rotated_corners = np.dot(robot_corners, rotation_matrix.T)
    translated_corners = rotated_corners + [rx, ry]
    corners = translated_corners.astype(int)
    cv2.polylines(lmap, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw robot orientation and predicted pose
    end_x = int(rx + 30 * math.cos(rtheta))
    end_y = int(ry + 30 * math.sin(rtheta))
    cv2.line(lmap, (int(rx), int(ry)), (end_x, end_y), (255, 0, 0), 2)
    
    pred_x, pred_y, pred_theta = predicted_pose(particles, weights)
    end_pred_x = int(pred_x + 30 * math.cos(pred_theta))
    end_pred_y = int(pred_y + 30 * math.sin(pred_theta))
    cv2.line(lmap, (int(pred_x), int(pred_y)), (end_pred_x, end_pred_y), (255, 0, 255), 2)
    
    # Draw waypoints and trajectory
    for wp in waypoints:
        cv2.circle(lmap, (int(wp[0]), int(wp[1])), 5, (0, 255, 255), -1)
    
    if len(trajectory) > 1:
        for i in range(len(trajectory) - 1):
            cv2.line(lmap, (int(trajectory[i][0]), int(trajectory[i][1])),
                     (int(trajectory[i + 1][0]), int(trajectory[i + 1][1])), (0, 0, 255), 2)
    
    return lmap

def move_robot(rx, ry, rtheta, forward, turn):
    rtheta += turn
    new_rx = rx + forward * math.cos(rtheta)
    new_ry = ry + forward * math.sin(rtheta)
    if map[int(new_ry), int(new_rx)] == 255:
        return new_rx, new_ry, rtheta
    return rx, ry, rtheta

def check_obstacle(rx, ry, angle, distance):
    for i in range(1, int(distance)):
        x = int(rx + i * math.cos(angle))
        y = int(ry + i * math.sin(angle))
        if 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT:
            if map[y, x] == 0:
                return True
    return False

# Main execution
init_particles()

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('simulation.mp4', fourcc, 20.0, (MAP_WIDTH, MAP_HEIGHT))

# Define waypoints
waypoints = [
    (50, 450),    # Start
    (150, 450),   # First intermediate
    (250, 310),   # Second intermediate
    (200, 100),   # Third intermediate
    (450, 20)     # End
]

trajectory = []
current_waypoint = 1
running = True

while running:
    frame = draw_map(rx, ry, rtheta, particles, waypoints, trajectory)
    cv2.imshow("Particle Filter Simulation", frame)
    out.write(frame)
    
    target_x, target_y = waypoints[current_waypoint]
    dx = target_x - rx
    dy = target_y - ry
    target_angle = math.atan2(dy, dx)
    distance_to_waypoint = math.sqrt(dx*dx + dy*dy)
    
    if distance_to_waypoint < 20:
        if current_waypoint < len(waypoints) - 1:
            current_waypoint += 1
        else:
            print("Final target reached!")
            break
    
    angle_diff = (target_angle - rtheta + math.pi) % (2 * math.pi) - math.pi
    turn = np.sign(angle_diff) * min(abs(angle_diff), TURN)
    forward = STEP
    
    if check_obstacle(rx, ry, rtheta, ROBOT_WIDTH * 2):
        left_clear = not check_obstacle(rx, ry, rtheta - math.pi/4, ROBOT_WIDTH * 2)
        right_clear = not check_obstacle(rx, ry, rtheta + math.pi/4, ROBOT_WIDTH * 2)
        
        if left_clear and right_clear:
            turn = TURN if angle_diff > 0 else -TURN
        elif left_clear:
            turn = -TURN
        elif right_clear:
            turn = TURN
        else:
            turn = TURN
        forward = STEP / 2
    
    rx, ry, rtheta = move_robot(rx, ry, rtheta, forward, turn)
    trajectory.append((rx, ry))
    motion_update(particles, forward, turn)
    
    rx = max(0, min(MAP_WIDTH - 1, rx))
    ry = max(0, min(MAP_HEIGHT - 1, ry))
    particles[:, 0] = np.clip(particles[:, 0], 0, MAP_WIDTH - 1)
    particles[:, 1] = np.clip(particles[:, 1], 0, MAP_HEIGHT - 1)
    
    weight_update(particles, rx, ry)
    particles = resample(particles, weights)
    
    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        running = False

out.release()
cv2.destroyAllWindows()
