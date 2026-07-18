import cv2
import mediapipe as mp
import numpy as np
import time
import collections

# ==========================================================================
# CONSTANTS & CONFIGURATION
# ==========================================================================
N = 4500  # Number of particles
fov_factor = 320  # Focal length for perspective projection
camera_dist = 6.0  # Camera distance along Z-axis
morph_speed = 0.04  # Speed of shape morphing
position_history_limit = 15
finger_history_limit = 15

# ==========================================================================
# STATE VARIABLES
# ==========================================================================
active_model = 'rose'  # 'rose', 'lotus', 'sunflower', 'tulip', 'lily'
gesture_state = 'NONE'  # 'OPEN', 'CLOSED', 'NONE'
scatter_amount = 0.0
target_scatter_amount = 0.0
current_box_scale = 1.0
target_box_scale = 1.0
base_rotation_y = 0.0

# Interpolation buffers
base_positions = np.zeros((N, 3), dtype=np.float32)
base_colors = np.zeros((N, 3), dtype=np.float32)

# Model buffers
rose_positions = np.zeros((N, 3), dtype=np.float32)
rose_colors = np.zeros((N, 3), dtype=np.float32)
lotus_positions = np.zeros((N, 3), dtype=np.float32)
lotus_colors = np.zeros((N, 3), dtype=np.float32)
sunflower_positions = np.zeros((N, 3), dtype=np.float32)
sunflower_colors = np.zeros((N, 3), dtype=np.float32)
tulip_positions = np.zeros((N, 3), dtype=np.float32)
tulip_colors = np.zeros((N, 3), dtype=np.float32)
lily_positions = np.zeros((N, 3), dtype=np.float32)
lily_colors = np.zeros((N, 3), dtype=np.float32)

# Bounding box corners (3D box of size 3.2 x 3.2 x 3.8)
box_corners = np.array([
    [-1.6, -1.6, -1.9],
    [ 1.6, -1.6, -1.9],
    [ 1.6,  1.6, -1.9],
    [-1.6,  1.6, -1.9],
    [-1.6, -1.6,  1.9],
    [ 1.6, -1.6,  1.9],
    [ 1.6,  1.6,  1.9],
    [-1.6,  1.6,  1.9]
], dtype=np.float32)

# Rolling history buffers
raw_position_history = collections.deque(maxlen=position_history_limit)
finger_count_history = collections.deque(maxlen=finger_history_limit)

# Target coordinates for the particle cloud center (lerped position)
target_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
smoothed_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# ==========================================================================
# PROCEDURAL 3D FLOWERS GENERATORS
# ==========================================================================
def generate_procedural_models():
    global base_positions, base_colors
    
    # ----------------------------------------------------------------------
    # 1. Vibrant Red Rose
    # ----------------------------------------------------------------------
    # 1a. Rose Stem (800 particles: 0 to 799)
    for i in range(800):
        y = -1.6 + (i / 799.0) * 1.2
        angle = (i * 0.2) % (np.pi * 2)
        r_stem = 0.045 + 0.01 * np.sin(y * 8.0)
        x = r_stem * np.cos(angle) + 0.06 * np.sin(y * 4.0)
        z = r_stem * np.sin(angle) + 0.06 * np.cos(y * 4.0)
        rose_positions[i] = [x, y, z]
        rose_colors[i] = [0.10, 0.52, 0.16]  # Leaf green

    # 1b. Rose Leaves (600 particles: 800 to 1399)
    leaf_p_idx = 800
    leaf_branches = [
        {"height": -1.0, "direction": 1, "rot": 0.2},
        {"height": -0.7, "direction": -1, "rot": -0.6}
    ]
    for branch in leaf_branches:
        num_p = 300
        for k in range(num_p):
            u = k / (num_p - 1.0)
            side = 1.0 if np.random.rand() > 0.5 else -1.0
            max_w = 0.24 * np.sin(u * np.pi)
            w = side * max_w * np.random.rand()
            
            lx = branch["direction"] * u * 0.58
            ly = branch["height"] + 0.12 * np.sin(u * np.pi) + 0.22 * u
            lz = w
            
            cos_r, sin_r = np.cos(branch["rot"]), np.sin(branch["rot"])
            rose_positions[leaf_p_idx] = [
                lx * cos_r - lz * sin_r,
                ly,
                lx * sin_r + lz * cos_r
            ]
            rose_colors[leaf_p_idx] = [0.08, 0.42, 0.12]
            leaf_p_idx += 1

    # 1c. Rose Petals (3100 particles: 1400 to 4499)
    rose_petal_idx = 1400
    rose_layers = 6
    for l in range(rose_layers):
        layer_p = 3100 // rose_layers
        R = 0.14 + l * 0.16
        H = 0.24 - l * 0.07
        num_petals = 4 + l
        rot_phase = l * 1.8
        
        for j in range(layer_p):
            if rose_petal_idx >= 4500:
                break
            u = np.random.rand()
            v = np.random.rand() * np.pi * 2
            
            petal_wave = np.sin(num_petals * v + rot_phase)
            r = R * u * (0.85 + 0.15 * petal_wave)
            x = r * np.cos(v)
            z = r * np.sin(v)
            
            y = H + 0.62 * (u ** 1.4) * (1.0 - 0.58 * (u ** 2.5)) + 0.05 * petal_wave
            rose_positions[rose_petal_idx] = [x, y, z]
            
            if l < 2:
                # Deep velvet red center core
                rose_colors[rose_petal_idx] = [0.42 + 0.15 * np.random.rand(), 0.01, 0.03]
            else:
                # Bright glowing outer red
                rose_colors[rose_petal_idx] = [0.88 + 0.12 * u, 0.02 + 0.08 * (1.0 - u), 0.06 + 0.08 * u]
            rose_petal_idx += 1

    # ----------------------------------------------------------------------
    # 2. Sacred Pink Lotus
    # ----------------------------------------------------------------------
    # 2a. Floating Lily Pad (1000 particles: 0 to 999)
    pad_idx = 0
    while pad_idx < 1000:
        r = 1.4 * np.sqrt(np.random.rand())
        theta = np.random.rand() * np.pi * 2
        
        is_in_notch = (theta < 0.28) or (theta > (np.pi * 2 - 0.28))
        if is_in_notch and r > 0.15:
            continue
        
        lotus_positions[pad_idx] = [r * np.cos(theta), -0.42, r * np.sin(theta)]
        lotus_colors[pad_idx] = [0.05 + 0.05 * r, 0.48 + 0.08 * (1.0 - r), 0.15]
        pad_idx += 1

    # 2b. Pointed Lotus Petals (3000 particles: 1000 to 3999)
    lotus_petal_idx = 1000
    lotus_layers = 5
    for l in range(lotus_layers):
        layer_p = 3000 // lotus_layers
        R = 0.18 + l * 0.22
        H = 0.18 - l * 0.08
        num_petals = 6 + l * 2
        rot_phase = l * 1.5
        
        for j in range(layer_p):
            if lotus_petal_idx >= 4000:
                break
            u = np.random.rand()
            v = np.random.rand() * np.pi * 2
            
            pointed_wave = 1.0 - 0.38 * np.abs(np.sin((num_petals * v + rot_phase) / 2.0))
            r = R * u * pointed_wave
            x = r * np.cos(v)
            z = r * np.sin(v)
            y = H + 0.38 * np.sin(u * np.pi / 2.0) + 0.04 * np.sin(num_petals * v)
            
            lotus_positions[lotus_petal_idx] = [x, y, z]
            
            pink_factor = u * 0.78 + (l / (lotus_layers - 1.0)) * 0.22
            lotus_colors[lotus_petal_idx] = [1.0, 0.90 - pink_factor * 0.58, 0.94 - pink_factor * 0.28]
            lotus_petal_idx += 1

    # 2c. Golden Stamen (500 particles: 4000 to 4499)
    for i in range(4000, 4500):
        r = 0.16 * np.sqrt(np.random.rand())
        theta = np.random.rand() * np.pi * 2
        lotus_positions[i] = [r * np.cos(theta), 0.18 + (np.random.rand() - 0.5) * 0.05, r * np.sin(theta)]
        lotus_colors[i] = [1.0, 0.76, 0.0]

    # ----------------------------------------------------------------------
    # 3. Golden Sunflower
    # ----------------------------------------------------------------------
    # 3a. Sunflower Stem (800 particles: 0 to 799)
    for i in range(800):
        y = -1.6 + (i / 799.0) * 1.1
        angle = (i * 0.15) % (np.pi * 2)
        r_stem = 0.055
        sunflower_positions[i] = [r_stem * np.cos(angle), y, r_stem * np.sin(angle)]
        sunflower_colors[i] = [0.12, 0.48, 0.14]

    # 3b. Fibonacci Seed Disk (1700 particles: 800 to 2499)
    for i in range(800, 2500):
        idx = i - 800
        theta = idx * 137.5 * (np.pi / 180.0)
        r = 0.65 * np.sqrt(idx / 1700.0)
        y = -0.05 + 0.08 * (r * r)
        sunflower_positions[i] = [r * np.cos(theta), y, r * np.sin(theta)]
        ratio = idx / 1700.0
        sunflower_colors[i] = [0.14 + ratio * 0.44, 0.08 + ratio * 0.32, 0.02 + ratio * 0.04]

    # 3c. Double-Layer Sunflower Petals (2000 particles: 2500 to 4499)
    sf_petal_idx = 2500
    num_sf_petals = 34
    petals_per_ring = 1000
    for ring in range(2):
        R_base = 0.62 + ring * 0.05
        for j in range(petals_per_ring):
            if sf_petal_idx >= 4500:
                break
            u = np.random.rand()
            v = np.random.rand() * np.pi * 2
            
            petal_wave = np.sin(num_sf_petals * v + ring * np.pi)
            r = R_base + u * 0.66
            x = r * np.cos(v + petal_wave * 0.025)
            z = r * np.sin(v + petal_wave * 0.025)
            y = -0.04 - 0.08 * u + 0.015 * petal_wave
            
            sunflower_positions[sf_petal_idx] = [x, y, z]
            sunflower_colors[sf_petal_idx] = [1.0, 0.82 + 0.12 * u, 0.0]
            sf_petal_idx += 1

    # ----------------------------------------------------------------------
    # 4. Sunset Orange Tulip
    # ----------------------------------------------------------------------
    # 4a. Tulip Stem (600 particles: 0 to 599)
    for i in range(600):
        y = -1.6 + (i / 599.0) * 1.5
        angle = (i * 0.25) % (np.pi * 2)
        r_stem = 0.05
        tulip_positions[i] = [r_stem * np.cos(angle), y, r_stem * np.sin(angle)]
        tulip_colors[i] = [0.12, 0.52, 0.15]
    # 4b. Tulip Leaves (400 particles: 600 to 999)
    tulip_leaf_idx = 600
    for l_idx, dir_val in enumerate([1, -1]):
        num_p = 200
        rot = l_idx * np.pi + 0.3
        for k in range(num_p):
            u = k / (num_p - 1.0)
            side = 1.0 if np.random.rand() > 0.5 else -1.0
            max_w = 0.28 * np.sin(u * np.pi)
            w = side * max_w * np.random.rand()
            
            lx = dir_val * u * 0.46
            ly = -1.1 + u * 0.85 + 0.08 * np.sin(u * np.pi)
            lz = w
            
            cos_r, sin_r = np.cos(rot), np.sin(rot)
            tulip_positions[tulip_leaf_idx] = [
                lx * cos_r - lz * sin_r,
                ly,
                lx * sin_r + lz * cos_r
            ]
            tulip_colors[tulip_leaf_idx] = [0.15, 0.48, 0.12]
            tulip_leaf_idx += 1
    # 4c. Tulip Cup Petals (3500 particles: 1000 to 4499)
    tulip_petal_idx = 1000
    tulip_petal_count = 6
    p_per_petal = 3500 // tulip_petal_count
    for p in range(tulip_petal_count):
        is_outer = p >= 3
        base_angle = (p % 3) * (np.pi * 2.0 / 3.0) + (np.pi / 3.0 if is_outer else 0.0)
        R = 0.36 if is_outer else 0.32
        for j in range(p_per_petal):
            if tulip_petal_idx >= 4500:
                break
            u = np.random.rand()
            theta = (np.random.rand() - 0.5) * 1.35
            abs_angle = base_angle + theta * 0.55
            
            curve_in = 1.0 - 0.25 * ((u - 0.72) ** 2.0) - 0.3 * (u ** 3.0)
            r = R * curve_in
            x = r * np.cos(abs_angle)
            z = r * np.sin(abs_angle)
            y = -0.08 + u * 0.78 - 0.15 * np.cos(theta) * (1.0 - u)
            
            tulip_positions[tulip_petal_idx] = [x, y, z]
            tulip_colors[tulip_petal_idx] = [1.0, 0.12 + u * 0.74, 0.05]
            tulip_petal_idx += 1

    # ----------------------------------------------------------------------
    # 5. Stargazer Pink Lily
    # ----------------------------------------------------------------------
    # 5a. Lily Stem (500 particles: 0 to 499)
    for i in range(500):
        y = -1.6 + (i / 499.0) * 1.5
        angle = (i * 0.3) % (np.pi * 2)
        r_stem = 0.045
        lily_positions[i] = [r_stem * np.cos(angle), y, r_stem * np.sin(angle)]
        lily_colors[i] = [0.10, 0.50, 0.12]
    # Lily Whorl Leaves (300 particles: 500 to 799)
    lily_leaf_idx = 500
    for w_idx in range(3):
        leaf_y = -1.2 + w_idx * 0.4
        num_whorl = 4
        p_per_whorl_leaf = 300 // (3 * num_whorl)
        for l in range(num_whorl):
            leaf_angle = (l * np.pi * 2.0 / num_whorl) + w_idx * 0.5
            for j in range(p_per_whorl_leaf):
                if lily_leaf_idx >= 800:
                    break
                u = j / (p_per_whorl_leaf - 1.0)
                side = 1.0 if np.random.rand() > 0.5 else -1.0
                max_w = 0.08 * np.sin(u * np.pi)
                w = side * max_w * np.random.rand()
                
                dist = 0.1 + u * 0.52
                lx = dist * np.cos(leaf_angle) + w * np.sin(leaf_angle)
                lz = dist * np.sin(leaf_angle) - w * np.cos(leaf_angle)
                ly = leaf_y - 0.12 * (u ** 2.0)
                
                lily_positions[lily_leaf_idx] = [lx, ly, lz]
                lily_colors[lily_leaf_idx] = [0.08, 0.44, 0.10]
                lily_leaf_idx += 1
    # 5b. Trumpet Lily Petals (3200 particles: 800 to 3999)
    lily_petal_idx = 800
    num_lily_petals = 6
    p_per_lily_petal = 3200 // num_lily_petals
    for p in range(num_lily_petals):
        base_angle = p * (np.pi * 2.0 / num_lily_petals)
        for j in range(p_per_lily_petal):
            if lily_petal_idx >= 4000:
                break
            u = np.random.rand()
            theta = (np.random.rand() - 0.5) * 0.72
            abs_angle = base_angle + theta * 0.48 * (1.0 - 0.4 * u)
            rad_dist = 0.18 + u * 1.25
            
            x = rad_dist * np.cos(abs_angle)
            z = rad_dist * np.sin(abs_angle)
            y = 0.12 - 0.65 * (u ** 1.8) + 0.12 * np.sin(u * np.pi)
            
            lily_positions[lily_petal_idx] = [x, y, z]
            
            is_stripe = 1.0 - np.abs(theta / 0.36)
            stripe_strength = np.maximum(0.0, is_stripe) * (0.2 + u * 0.8)
            lily_colors[lily_petal_idx] = [1.0, 0.94 - stripe_strength * 0.62, 0.96 - stripe_strength * 0.18]
            lily_petal_idx += 1
    # 5c. Lily Stamens (500 particles: 4000 to 4499)
    lily_stamen_idx = 4000
    num_stamens = 6
    p_per_stamen = 500 // num_stamens
    for s in range(num_stamens):
        stamen_angle = s * (np.pi * 2.0 / num_stamens) + 0.5
        for j in range(p_per_stamen):
            if lily_stamen_idx >= 4500:
                break
            u = j / (p_per_stamen - 1.0)
            r = u * 0.44
            x = r * np.cos(stamen_angle)
            z = r * np.sin(stamen_angle)
            y = 0.1 + u * 0.46 - 0.18 * (u ** 2.0)
            
            lily_positions[lily_stamen_idx] = [x, y, z]
            
            if u > 0.94:
                # Anthers (Pollen Pods)
                lily_colors[lily_stamen_idx] = [0.72, 0.22, 0.05]
            else:
                # Filaments
                lily_colors[lily_stamen_idx] = [0.55, 0.88, 0.42]
            lily_stamen_idx += 1

    # Initialize current render values to Rose
    base_positions[:] = rose_positions
    base_colors[:] = rose_colors

# ==========================================================================
# SMOOTH GESTURE STATE AVERAGING
# ==========================================================================
def get_smoothed_target(rx, ry, rz):
    raw_position_history.append((rx, ry, rz))
    avg = np.mean(raw_position_history, axis=0)
    return avg[0], avg[1], avg[2]

# ==========================================================================
# 3D PERSPECTIVE PROJECTION ENGINE
# ==========================================================================
def project_points(points, center_x, center_y, angle_x, angle_y, scale):
    # Rotate around Y axis
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    x = points[:, 0] * scale
    y = points[:, 1] * scale
    z = points[:, 2] * scale
    
    x_rot = x * cos_y + z * sin_y
    z_rot = -x * sin_y + z * cos_y
    
    # Rotate around X axis
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    y_rot = y * cos_x - z_rot * sin_x
    z_rot = y * sin_x + z_rot * cos_x
    
    # Perspective projection calculation
    z_proj = z_rot + camera_dist
    z_proj = np.maximum(z_proj, 0.1)  # prevent divide by zero
    
    xs = (center_x + x_rot * fov_factor / z_proj).astype(np.int32)
    ys = (center_y - y_rot * fov_factor / z_proj).astype(np.int32)
    
    return xs, ys, z_rot

# ==========================================================================
# MAIN EVENT LOOP
# ==========================================================================
def main():
    global active_model, gesture_state, scatter_amount, target_scatter_amount
    global current_box_scale, target_box_scale, base_rotation_y, base_positions, base_colors
    global smoothed_position
    
    # Generate models
    print("Generating flower geometries...")
    generate_procedural_models()
    
    # Initialize OpenCV capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set standard capture size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55
    )
    
    print("Showcase started. Lock position: Right side of screen.")
    print("Controls: Open hand (scatter), Closed hand (morph).")
    print("Press 'q' or 'ESC' on the OpenCV window to exit.")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror frame to match typical webcam view
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # Calculate centers
        center_x = int(width * 0.72)
        center_y = int(height * 0.5)
        
        # Process Hand Tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        is_tracking = False
        detected_gesture = 'NONE'
        
        if results.multi_hand_landmarks:
            is_tracking = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Map wrist and middle finger MCP
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]
            
            # Detect finger extension (identical logic to JS implementation)
            def is_extended(tip_idx, mcp_idx):
                tip = hand_landmarks.landmark[tip_idx]
                mcp = hand_landmarks.landmark[mcp_idx]
                d_tip = np.hypot(np.hypot(tip.x - wrist.x, tip.y - wrist.y), tip.z - wrist.z)
                d_mcp = np.hypot(np.hypot(mcp.x - wrist.x, mcp.y - wrist.y), mcp.z - wrist.z)
                return d_tip > d_mcp

            open_fingers = sum([
                is_extended(8, 5),   # Index
                is_extended(12, 9),  # Middle
                is_extended(16, 13), # Ring
                is_extended(20, 17)  # Pinky
            ])
            
            # Smoothed sliding filter for gestures
            finger_count_history.append(open_fingers)
            avg_fingers = np.mean(finger_count_history)
            
            if avg_fingers >= 3.0:
                detected_gesture = 'OPEN'
            elif avg_fingers <= 0.8:
                detected_gesture = 'CLOSED'
            else:
                detected_gesture = gesture_state
                
            # Raw translation coordinates (relative mapping for tilt rotations)
            raw_x = (0.5 - wrist.x) * 6.5
            raw_y = (0.5 - wrist.y) * 4.5
            raw_z = (wrist.z + 0.1) * 8.0
            
            # Smooth coordinates
            sm_x, sm_y, sm_z = get_smoothed_target(raw_x, raw_y, raw_z)
            target_position[0] = sm_x
            target_position[1] = sm_y
            target_position[2] = sm_z
        else:
            if gesture_state != 'NONE':
                gesture_state = 'NONE'
                target_scatter_amount = 0.0
                target_box_scale = 1.0
                finger_count_history.clear()
                raw_position_history.clear()
            target_position.fill(0.0)
            
        # Smooth coordinate translation (Layer 2)
        smoothed_position += (target_position - smoothed_position) * 0.06
        
        # Handle Gestures (Scatter / Morphing trigger)
        if detected_gesture != gesture_state and detected_gesture != 'NONE':
            if detected_gesture == 'OPEN':
                target_scatter_amount = 2.4
                target_box_scale = 1.35
            elif detected_gesture == 'CLOSED':
                if active_model == 'rose':
                    active_model = 'lotus'
                elif active_model == 'lotus':
                    active_model = 'sunflower'
                elif active_model == 'sunflower':
                    active_model = 'tulip'
                elif active_model == 'tulip':
                    active_model = 'lily'
                else:
                    active_model = 'rose'
                
                target_scatter_amount = 0.0
                target_box_scale = 1.0
            gesture_state = detected_gesture
            
        # Slower scatter transitions
        scatter_amount += (target_scatter_amount - scatter_amount) * 0.045
        
        # Bounding box scaling
        current_box_scale += (target_box_scale - current_box_scale) * 0.08
        
        # Target positions selection
        if active_model == 'rose':
            tx_pos, tx_col = rose_positions, rose_colors
        elif active_model == 'lotus':
            tx_pos, tx_col = lotus_positions, lotus_colors
        elif active_model == 'sunflower':
            tx_pos, tx_col = sunflower_positions, sunflower_colors
        elif active_model == 'tulip':
            tx_pos, tx_col = tulip_positions, tulip_colors
        else:
            tx_pos, tx_col = lily_positions, lily_colors
            
        # Smooth coordinate and color morphing
        base_positions += (tx_pos - base_positions) * morph_speed
        base_colors += (tx_col - base_colors) * morph_speed
        
        # Apply scattering physics
        if scatter_amount > 0.02:
            time_scale = time.time() * 2.2
            expansion = 1.0 + scatter_amount * 0.85
            
            # Vectorized noise offsets
            indices = np.arange(N)
            noise_x = np.sin(time_scale + indices * 0.17) * scatter_amount * 0.4
            noise_y = np.cos(time_scale * 0.8 + indices * 0.23) * scatter_amount * 0.4
            noise_z = np.sin(time_scale * 1.3 - indices * 0.13) * scatter_amount * 0.4
            
            render_positions = base_positions * expansion
            render_positions[:, 0] += noise_x
            render_positions[:, 1] += noise_y
            render_positions[:, 2] += noise_z
        else:
            render_positions = base_positions.copy()
            
        # Calculate Rotations (Automatic base rotation + hand tilt offset)
        base_rotation_y += 0.005
        angle_x = smoothed_position[1] * 0.12
        angle_y = base_rotation_y + smoothed_position[0] * 0.12
        
        # Perspective projection of particles
        xs, ys, zs = project_points(render_positions, center_x, center_y, angle_x, angle_y, 1.0)
        
        # Perspective projection of bounding box corners
        c_xs, c_ys, c_zs = project_points(box_corners, center_x, center_y, angle_x, angle_y, current_box_scale)
        
        # Depth sorting (Painter's Algorithm) to render correct overlapping depth
        depth_indices = np.argsort(zs)[::-1]  # sort furthest to nearest
        xs_sorted = xs[depth_indices]
        ys_sorted = ys[depth_indices]
        col_sorted = base_colors[depth_indices]
        
        # Draw 3D Ground Grid lines beneath the right-aligned model
        grid_lines_3d = []
        for x_val in np.linspace(-3.0, 3.0, 7):
            grid_lines_3d.append([x_val, -2.5, -3.0])
            grid_lines_3d.append([x_val, -2.5,  3.0])
        for z_val in np.linspace(-3.0, 3.0, 7):
            grid_lines_3d.append([-3.0, -2.5, z_val])
            grid_lines_3d.append([ 3.0, -2.5, z_val])
        grid_lines_3d = np.array(grid_lines_3d, dtype=np.float32)
        grid_lines_3d[:, 0] += 1.6
        
        g_xs, g_ys, _ = project_points(grid_lines_3d, center_x, center_y, angle_x, angle_y, 1.0)
        for g_idx in range(0, len(g_xs), 2):
            cv2.line(frame, (g_xs[g_idx], g_ys[g_idx]), (g_xs[g_idx+1], g_ys[g_idx+1]), (120, 60, 40), 1)

        # Draw Bounding Box wireframe edges
        box_edges = [
            # Bottom Loop
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top Loop
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical Pillars
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Get active model color profile for bounding box lines
        box_color_bgr = (254, 242, 0)
        if gesture_state == 'OPEN':
            box_color_bgr = (126, 71, 255)
        else:
            if active_model == 'rose': box_color_bgr = (126, 71, 255)
            elif active_model == 'lotus': box_color_bgr = (162, 133, 255)
            elif active_model == 'sunflower': box_color_bgr = (3, 183, 255)
            elif active_model == 'tulip': box_color_bgr = (3, 125, 255)
            elif active_model == 'lily': box_color_bgr = (166, 112, 255)

        for edge in box_edges:
            p1, p2 = edge
            cv2.line(frame, (c_xs[p1], c_ys[p1]), (c_xs[p2], c_ys[p2]), box_color_bgr, 1, cv2.LINE_AA)
            
        # Draw Corner Marker Spheres
        for c_idx in range(8):
            cv2.circle(frame, (c_xs[c_idx], c_ys[c_idx]), 4, box_color_bgr, -1, cv2.LINE_AA)
            
        # Draw 3D Particles
        for i in range(N):
            px, py = xs_sorted[i], ys_sorted[i]
            if 0 <= px < width and 0 <= py < height:
                b = int(col_sorted[i, 2] * 255)
                g = int(col_sorted[i, 1] * 255)
                r = int(col_sorted[i, 0] * 255)
                cv2.circle(frame, (px, py), 2, (b, g, r), -1)

        # ----------------------------------------------------------------------
        # HUD OVERLAYS
        # ----------------------------------------------------------------------
        hud_x = int(width * 0.65)
        hud_y_start = int(height * 0.12)
        
        cv2.rectangle(frame, (hud_x - 15, hud_y_start - 30), (width - 40, hud_y_start + 110), (12, 10, 8), -1)
        cv2.rectangle(frame, (hud_x - 15, hud_y_start - 30), (width - 40, hud_y_start + 110), (45, 40, 35), 1)
        
        # 1. Model Label
        model_names = {
            'rose': ('VIBRANT RED ROSE', (126, 71, 255)),
            'lotus': ('SACRED PINK LOTUS', (162, 133, 255)),
            'sunflower': ('GOLDEN SUNFLOWER', (3, 183, 255)),
            'tulip': ('SUNSET ORANGE TULIP', (3, 125, 255)),
            'lily': ('STARGAZER PINK LILY', (166, 112, 255))
        }
        lbl_text, lbl_color = model_names[active_model]
        cv2.putText(frame, "ACTIVE MODEL:", (hud_x, hud_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 150, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, lbl_text, (hud_x, hud_y_start + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, lbl_color, 2, cv2.LINE_AA)
        
        # 2. Gesture Label
        gesture_lbl = "ASSEMBLED"
        gesture_col = (160, 214, 6)
        if gesture_state == 'OPEN':
            gesture_lbl = "SCATTERED"
            gesture_col = (126, 71, 255)
        elif gesture_state == 'NONE':
            gesture_lbl = "WAITING..."
            gesture_col = (160, 150, 140)
            
        cv2.putText(frame, "GESTURE STATE:", (hud_x, hud_y_start + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 150, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, gesture_lbl, (hud_x, hud_y_start + 77), cv2.FONT_HERSHEY_SIMPLEX, 0.58, gesture_col, 2, cv2.LINE_AA)
        
        # 3. Hand Status
        status_text = "HAND TRACKED" if is_tracking else "SEARCHING FOR HAND..."
        status_col = (6, 214, 160) if is_tracking else (140, 150, 160)
        cv2.putText(frame, status_text, (hud_x, hud_y_start + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_col, 1, cv2.LINE_AA)
        
        # Compute & Display FPS
        curr_time = time.time()
        fps = int(1.0 / max(0.001, curr_time - prev_time))
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (6, 214, 160), 2, cv2.LINE_AA)
        
        # Display frame
        cv2.imshow("3D Holographic Flower Interactive Showcase", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main()
