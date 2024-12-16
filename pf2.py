import cv2
import numpy as np
import math

# Map dimensions
MAP_HEIGHT, MAP_WIDTH = 500, 500
map = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8) * 255

# Robot initialization
rx, ry, rtheta = 250, 250, 0  # Robot's initial position (x, y, orientation)
STEP = 10                    # Movement step size
TURN = math.radians(15)      # Turn angle in radians
ROBOT_WIDTH, ROBOT_HEIGHT = 50, 30  # Dimensions of the robot block

# Particle filter parameters
NUM_PARTICLES = 1000
particles = None
weights = None

# Initialize particles
def init_particles():
    global particles, weights
    particles = np.zeros((NUM_PARTICLES, 3))
    particles[:, 0] = np.random.uniform(0, MAP_WIDTH, NUM_PARTICLES)  # X positions
    particles[:, 1] = np.random.uniform(0, MAP_HEIGHT, NUM_PARTICLES) # Y positions
    particles[:, 2] = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)  # Orientations
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

# Move particles based on robot motion
def motion_update(particles, forward, turn):
    noise_fwd = np.random.normal(0, 2, NUM_PARTICLES)  # Forward noise
    noise_turn = np.random.normal(0, 0.1, NUM_PARTICLES)  # Turn noise
    particles[:, 2] += turn + noise_turn
    particles[:, 0] += (forward + noise_fwd) * np.cos(particles[:, 2])
    particles[:, 1] += (forward + noise_fwd) * np.sin(particles[:, 2])

# Update particle weights based on sensor model
def weight_update(particles, rx, ry):
    global weights
    distances = np.sqrt((particles[:, 0] - rx)**2 + (particles[:, 1] - ry)**2)
    weights = np.exp(-distances / 60)  # Gaussian-like weight based on distance
    weights += 1e-300  # Avoid zero weights
    weights /= np.sum(weights)  # Normalize

# Resample particles based on weights
def resample(particles, weights):
    indices = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES, p=weights)
    particles = particles[indices]
    return particles

# Compute predicted pose from particles
def predicted_pose(particles, weights):
    mean_x = np.average(particles[:, 0], weights=weights)
    mean_y = np.average(particles[:, 1], weights=weights)
    mean_theta = np.arctan2(
        np.average(np.sin(particles[:, 2]), weights=weights),
        np.average(np.cos(particles[:, 2]), weights=weights)
    )
    return mean_x, mean_y, mean_theta

# Draw the map with robot and particles
def draw_map(rx, ry, rtheta, particles):
    lmap = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)  # Convert to color

    # Draw particles
    for px, py, _ in particles:
        cv2.circle(lmap, (int(px), int(py)), 2, (0, 0, 255), -1)

    # Draw robot as a rectangle with orientation
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

    # Draw robot's actual orientation
    end_x = int(rx + 50 * math.cos(rtheta))
    end_y = int(ry + 50 * math.sin(rtheta))
    cv2.line(lmap, (int(rx), int(ry)), (end_x, end_y), (255, 0, 0), 2)

    # Predicted pose
    pred_x, pred_y, pred_theta = predicted_pose(particles, weights)
    end_pred_x = int(pred_x + 50 * math.cos(pred_theta))
    end_pred_y = int(pred_y + 50 * math.sin(pred_theta))
    cv2.line(lmap, (int(pred_x), int(pred_y)), (end_pred_x, end_pred_y), (255, 0, 255), 2)

    cv2.imshow("Particle Filter Simulation", lmap)

# Robot movement
def move_robot(rx, ry, rtheta, forward, turn):
    rtheta += turn
    rx += forward * math.cos(rtheta)
    ry += forward * math.sin(rtheta)
    return rx, ry, rtheta

# Main interactive loop
init_particles()
cv2.namedWindow("Particle Filter Simulation")
cv2.setWindowProperty("Particle Filter Simulation", cv2.WND_PROP_TOPMOST, 1)

while True:
    draw_map(rx, ry, rtheta, particles)

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    if key == ord('w'):  # Move forward
        rx, ry, rtheta = move_robot(rx, ry, rtheta, STEP, 0)
        motion_update(particles, STEP, 0)
    elif key == ord('a'):  # Turn left
        rx, ry, rtheta = move_robot(rx, ry, rtheta, 0, -TURN)
        motion_update(particles, 0, -TURN)
    elif key == ord('d'):  # Turn right
        rx, ry, rtheta = move_robot(rx, ry, rtheta, 0, TURN)
        motion_update(particles, 0, TURN)
    elif key == ord('s'):  # Move backward
        rx, ry, rtheta = move_robot(rx, ry, rtheta, -STEP, 0)
        motion_update(particles, -STEP, 0)
    elif key == ord('q'):  # Quit
        print("Exiting simulation.")
        break

    # Ensure robot and particles stay within bounds
    rx = max(0, min(MAP_WIDTH - 1, rx))
    ry = max(0, min(MAP_HEIGHT - 1, ry))
    particles[:, 0] = np.clip(particles[:, 0], 0, MAP_WIDTH - 1)
    particles[:, 1] = np.clip(particles[:, 1], 0, MAP_HEIGHT - 1)

    # Particle filter steps
    weight_update(particles, rx, ry)
    particles = resample(particles, weights)

    # Close the simulation if the window is manually closed
    if cv2.getWindowProperty("Particle Filter Simulation", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed. Exiting simulation.")
        break

cv2.destroyAllWindows()
