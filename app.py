import cv2
import mediapipe as mp
import numpy as np
import time
import collections
import streamlit as st

# ==========================================================================
# STREAMLIT PAGE CONFIG
# ==========================================================================
st.set_page_config(
    page_title="3D Holographic Flower Showcase",
    page_icon="🌹",
    layout="wide"
)

# ==========================================================================
# CONSTANTS & CONFIGURATION
# ==========================================================================
BACKEND_URL = "https://tugl24abp9.execute-api.us-east-1.amazonaws.com/bloom"
N = 4500  # Number of particles
fov_factor = 320  # Focal length for perspective projection
camera_dist = 6.0  # Camera distance along Z-axis
morph_speed = 0.04  # Speed of shape morphing
position_history_limit = 15
finger_history_limit = 15

# ==========================================================================
# CACHED MODEL GENERATION (runs once per session, not on every rerun)
# ==========================================================================
@st.cache_resource(show_spinner="Generating procedural flower geometries...")
def build_models():
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

    # ----------------------------------------------------------------------
    # 1. Vibrant Red Rose
    # ----------------------------------------------------------------------
    for i in range(800):
        y = -1.6 + (i / 799.0) * 1.2
        angle = (i * 0.2) % (np.pi * 2)
        r_stem = 0.045 + 0.01 * np.sin(y * 8.0)
        x = r_stem * np.cos(angle) + 0.06 * np.sin(y * 4.0)
        z = r_stem * np.sin(angle) + 0.06 * np.cos(y * 4.0)
        rose_positions[i] = [x, y, z]
        rose_colors[i] = [0.10, 0.52, 0.16]

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
                rose_colors[rose_petal_idx] = [0.42 + 0.15 * np.random.rand(), 0.01, 0.03]
            else:
                rose_colors[rose_petal_idx] = [0.88 + 0.12 * u, 0.02 + 0.08 * (1.0 - u), 0.06 + 0.08 * u]
            rose_petal_idx += 1

    # ----------------------------------------------------------------------
    # 2. Sacred Pink Lotus
    # ----------------------------------------------------------------------
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

    for i in range(4000, 4500):
        r = 0.16 * np.sqrt(np.random.rand())
        theta = np.random.rand() * np.pi * 2
        lotus_positions[i] = [r * np.cos(theta), 0.18 + (np.random.rand() - 0.5) * 0.05, r * np.sin(theta)]
        lotus_colors[i] = [1.0, 0.76, 0.0]

    # ----------------------------------------------------------------------
    # 3. Golden Sunflower
    # ----------------------------------------------------------------------
    for i in range(800):
        y = -1.6 + (i / 799.0) * 1.1
        angle = (i * 0.15) % (np.pi * 2)
        r_stem = 0.055
        sunflower_positions[i] = [r_stem * np.cos(angle), y, r_stem * np.sin(angle)]
        sunflower_colors[i] = [0.12, 0.48, 0.14]

    for i in range(800, 2500):
        idx = i - 800
        theta = idx * 137.5 * (np.pi / 180.0)
        r = 0.65 * np.sqrt(idx / 1700.0)
        y = -0.05 + 0.08 * (r * r)
        sunflower_positions[i] = [r * np.cos(theta), y, r * np.sin(theta)]
        ratio = idx / 1700.0
        sunflower_colors[i] = [0.14 + ratio * 0.44, 0.08 + ratio * 0.32, 0.02 + ratio * 0.04]

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
    for i in range(600):
        y = -1.6 + (i / 599.0) * 1.5
        angle = (i * 0.25) % (np.pi * 2)
        r_stem = 0.05
        tulip_positions[i] = [r_stem * np.cos(angle), y, r_stem * np.sin(angle)]
        tulip_colors[i] = [0.12, 0.52, 0.15]

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
    for i in range(500):
        y = -1.6 + (i / 499.0) * 1.5
        angle = (i * 0.3) % (np.pi * 2)
        r_stem = 0.045
        lily_positions[i] = [r_stem * np.cos(angle), y, r_stem * np.sin(angle)]
        lily_colors[i] = [0.10, 0.50, 0.12]

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
                lily_colors[lily_stamen_idx] = [0.72, 0.22, 0.05]
            else:
                lily_colors[lily_stamen_idx] = [0.55, 0.88, 0.42]
            lily_stamen_idx += 1

    return {
        "rose": (rose_positions, rose_colors),
        "lotus": (lotus_positions, lotus_colors),
        "sunflower": (sunflower_positions, sunflower_colors),
        "tulip": (tulip_positions, tulip_colors),
        "lily": (lily_positions, lily_colors),
    }

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

box_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

model_names = {
    'rose': ('VIBRANT RED ROSE', (126, 71, 255)),
    'lotus': ('SACRED PINK LOTUS', (162, 133, 255)),
    'sunflower': ('GOLDEN SUNFLOWER', (3, 183, 255)),
    'tulip': ('SUNSET ORANGE TULIP', (3, 125, 255)),
    'lily': ('STARGAZER PINK LILY', (166, 112, 255))
}
model_order = ['rose', 'lotus', 'sunflower', 'tulip', 'lily']

# ==========================================================================
# 3D PERSPECTIVE PROJECTION ENGINE
# ==========================================================================
def project_points(points, center_x, center_y, angle_x, angle_y, scale):
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    x = points[:, 0] * scale
    y = points[:, 1] * scale
    z = points[:, 2] * scale

    x_rot = x * cos_y + z * sin_y
    z_rot = -x * sin_y + z * cos_y

    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    y_rot = y * cos_x - z_rot * sin_x
    z_rot = y * sin_x + z_rot * cos_x

    z_proj = z_rot + camera_dist
    z_proj = np.maximum(z_proj, 0.1)

    xs = (center_x + x_rot * fov_factor / z_proj).astype(np.int32)
    ys = (center_y - y_rot * fov_factor / z_proj).astype(np.int32)

    return xs, ys, z_rot

# ==========================================================================
# SESSION STATE INITIALIZATION (persists across Streamlit reruns)
# ==========================================================================
def init_state():
    ss = st.session_state
    ss.setdefault("active_model", "rose")
    ss.setdefault("gesture_state", "NONE")
    ss.setdefault("scatter_amount", 0.0)
    ss.setdefault("target_scatter_amount", 0.0)
    ss.setdefault("current_box_scale", 1.0)
    ss.setdefault("target_box_scale", 1.0)
    ss.setdefault("base_rotation_y", 0.0)
    ss.setdefault("base_positions", None)
    ss.setdefault("base_colors", None)
    ss.setdefault("smoothed_position", np.zeros(3, dtype=np.float32))
    ss.setdefault("target_position", np.zeros(3, dtype=np.float32))
    ss.setdefault("raw_position_history", collections.deque(maxlen=position_history_limit))
    ss.setdefault("finger_count_history", collections.deque(maxlen=finger_history_limit))
    ss.setdefault("running", False)
    # Second-hand controls
    ss.setdefault("zoom_scale", 1.0)
    ss.setdefault("target_zoom_scale", 1.0)
    ss.setdefault("rot_offset", 0.0)
    ss.setdefault("target_rot_offset", 0.0)
    ss.setdefault("second_hand_tracking", False)

def get_smoothed_target(rx, ry, rz):
    hist = st.session_state.raw_position_history
    hist.append((rx, ry, rz))
    avg = np.mean(hist, axis=0)
    return avg[0], avg[1], avg[2]

# ==========================================================================
# STREAMLIT UI
# ==========================================================================
st.title("🌹 3D Holographic Flower Interactive Showcase")
st.caption("Hand-gesture controlled particle-cloud flowers, rendered live from your webcam feed.")

init_state()
models = build_models()
if st.session_state.base_positions is None:
    st.session_state.base_positions = models["rose"][0].copy()
    st.session_state.base_colors = models["rose"][1].copy()

with st.sidebar:
    st.header("Controls")
    run = st.checkbox("Start camera", value=st.session_state.running)
    st.session_state.running = run
    st.markdown("---")
    st.subheader("Primary hand (right side)")
    st.write("✋ **Open hand** — scatter the particles")
    st.write("✊ **Closed hand** — morph to the next flower")
    st.write("↕️ **Move hand** — tilt the flower")
    st.markdown("---")
    st.subheader("Second hand (left side)")
    st.write("⬆️⬇️ **Raise / lower** — zoom in / out")
    st.write("⬅️➡️ **Move left / right** — extra spin")
    st.caption("Show both hands to camera. The hand further right drives gestures; the other hand controls zoom & spin.")
    st.markdown("---")
    st.subheader("Jump to a flower")
    for m in model_order:
        label, color_bgr = model_names[m]
        if st.button(label.title(), key=f"jump_{m}", use_container_width=True):
            st.session_state.active_model = m
            st.session_state.target_scatter_amount = 0.0
            st.session_state.target_box_scale = 1.0
    st.markdown("---")
    status_placeholder = st.empty()

frame_placeholder = st.empty()
fps_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open the webcam. Check camera permissions and try again.")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55
        )

        prev_time = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Lost connection to the webcam.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            center_x = int(width * 0.72)
            center_y = int(height * 0.5)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            is_tracking = False
            detected_gesture = 'NONE'
            second_hand_landmarks = None

            hand_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []

            # Sort detected hands left-to-right by average landmark x.
            # The right-most hand is the PRIMARY (gesture/tilt) hand, since it
            # sits near the flower which renders on the right side of the frame.
            # A second hand further left becomes the SECONDARY (zoom/spin) hand.
            if hand_list:
                hand_list = sorted(
                    hand_list,
                    key=lambda hl: np.mean([lm.x for lm in hl.landmark])
                )
                primary_hand_landmarks = hand_list[-1]
                if len(hand_list) >= 2:
                    second_hand_landmarks = hand_list[0]
            else:
                primary_hand_landmarks = None

            if primary_hand_landmarks is not None:
                is_tracking = True
                hand_landmarks = primary_hand_landmarks

                wrist = hand_landmarks.landmark[0]

                def is_extended(tip_idx, mcp_idx):
                    tip = hand_landmarks.landmark[tip_idx]
                    mcp = hand_landmarks.landmark[mcp_idx]
                    d_tip = np.hypot(np.hypot(tip.x - wrist.x, tip.y - wrist.y), tip.z - wrist.z)
                    d_mcp = np.hypot(np.hypot(mcp.x - wrist.x, mcp.y - wrist.y), mcp.z - wrist.z)
                    return d_tip > d_mcp

                open_fingers = sum([
                    is_extended(8, 5),
                    is_extended(12, 9),
                    is_extended(16, 13),
                    is_extended(20, 17)
                ])

                st.session_state.finger_count_history.append(open_fingers)
                avg_fingers = np.mean(st.session_state.finger_count_history)

                if avg_fingers >= 3.0:
                    detected_gesture = 'OPEN'
                elif avg_fingers <= 0.8:
                    detected_gesture = 'CLOSED'
                else:
                    detected_gesture = st.session_state.gesture_state

                raw_x = (0.5 - wrist.x) * 6.5
                raw_y = (0.5 - wrist.y) * 4.5
                raw_z = (wrist.z + 0.1) * 8.0

                sm_x, sm_y, sm_z = get_smoothed_target(raw_x, raw_y, raw_z)
                st.session_state.target_position[0] = sm_x
                st.session_state.target_position[1] = sm_y
                st.session_state.target_position[2] = sm_z
            else:
                if st.session_state.gesture_state != 'NONE':
                    st.session_state.gesture_state = 'NONE'
                    st.session_state.target_scatter_amount = 0.0
                    st.session_state.target_box_scale = 1.0
                    st.session_state.finger_count_history.clear()
                    st.session_state.raw_position_history.clear()
                st.session_state.target_position[:] = 0.0

            # --- Secondary hand: zoom (vertical position) + extra spin (horizontal) ---
            if second_hand_landmarks is not None:
                st.session_state.second_hand_tracking = True
                sec_wrist = second_hand_landmarks.landmark[0]
                # Raise hand (small y) -> zoom in. Lower hand (large y) -> zoom out.
                st.session_state.target_zoom_scale = float(
                    np.clip(np.interp(sec_wrist.y, [0.15, 0.85], [1.8, 0.55]), 0.4, 2.2)
                )
                # Move hand left/right of center -> extra rotation offset.
                st.session_state.target_rot_offset = float((0.5 - sec_wrist.x) * 2.4)
            else:
                st.session_state.second_hand_tracking = False
                st.session_state.target_zoom_scale = 1.0
                st.session_state.target_rot_offset = 0.0

            st.session_state.smoothed_position += (
                st.session_state.target_position - st.session_state.smoothed_position
            ) * 0.06

            st.session_state.zoom_scale += (
                st.session_state.target_zoom_scale - st.session_state.zoom_scale
            ) * 0.08

            st.session_state.rot_offset += (
                st.session_state.target_rot_offset - st.session_state.rot_offset
            ) * 0.08

            if detected_gesture != st.session_state.gesture_state and detected_gesture != 'NONE':
                if detected_gesture == 'OPEN':
                    st.session_state.target_scatter_amount = 2.4
                    st.session_state.target_box_scale = 1.35
                elif detected_gesture == 'CLOSED':
                    idx = model_order.index(st.session_state.active_model)
                    st.session_state.active_model = model_order[(idx + 1) % len(model_order)]
                    st.session_state.target_scatter_amount = 0.0
                    st.session_state.target_box_scale = 1.0
                st.session_state.gesture_state = detected_gesture

            st.session_state.scatter_amount += (
                st.session_state.target_scatter_amount - st.session_state.scatter_amount
            ) * 0.045

            st.session_state.current_box_scale += (
                st.session_state.target_box_scale - st.session_state.current_box_scale
            ) * 0.08

            tx_pos, tx_col = models[st.session_state.active_model]

            st.session_state.base_positions += (tx_pos - st.session_state.base_positions) * morph_speed
            st.session_state.base_colors += (tx_col - st.session_state.base_colors) * morph_speed

            base_positions = st.session_state.base_positions
            base_colors = st.session_state.base_colors
            scatter_amount = st.session_state.scatter_amount

            if scatter_amount > 0.02:
                time_scale = time.time() * 2.2
                expansion = 1.0 + scatter_amount * 0.85

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

            st.session_state.base_rotation_y += 0.005
            angle_x = st.session_state.smoothed_position[1] * 0.12
            angle_y = (
                st.session_state.base_rotation_y
                + st.session_state.smoothed_position[0] * 0.12
                + st.session_state.rot_offset
            )
            zoom_scale = st.session_state.zoom_scale

            xs, ys, zs = project_points(render_positions, center_x, center_y, angle_x, angle_y, zoom_scale)
            c_xs, c_ys, c_zs = project_points(box_corners, center_x, center_y, angle_x, angle_y,
                                               st.session_state.current_box_scale * zoom_scale)

            depth_indices = np.argsort(zs)[::-1]
            xs_sorted = xs[depth_indices]
            ys_sorted = ys[depth_indices]
            col_sorted = base_colors[depth_indices]

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
                cv2.line(frame, (g_xs[g_idx], g_ys[g_idx]), (g_xs[g_idx + 1], g_ys[g_idx + 1]), (120, 60, 40), 1)

            box_color_bgr = (254, 242, 0)
            if st.session_state.gesture_state == 'OPEN':
                box_color_bgr = (126, 71, 255)
            else:
                box_color_bgr = model_names[st.session_state.active_model][1]

            for edge in box_edges:
                p1, p2 = edge
                cv2.line(frame, (c_xs[p1], c_ys[p1]), (c_xs[p2], c_ys[p2]), box_color_bgr, 1, cv2.LINE_AA)

            for c_idx in range(8):
                cv2.circle(frame, (c_xs[c_idx], c_ys[c_idx]), 4, box_color_bgr, -1, cv2.LINE_AA)

            for i in range(N):
                px, py = xs_sorted[i], ys_sorted[i]
                if 0 <= px < width and 0 <= py < height:
                    b = int(col_sorted[i, 2] * 255)
                    g = int(col_sorted[i, 1] * 255)
                    r = int(col_sorted[i, 0] * 255)
                    cv2.circle(frame, (px, py), 2, (b, g, r), -1)

            hud_x = int(width * 0.65)
            hud_y_start = int(height * 0.12)

            cv2.rectangle(frame, (hud_x - 15, hud_y_start - 30), (width - 40, hud_y_start + 140), (12, 10, 8), -1)
            cv2.rectangle(frame, (hud_x - 15, hud_y_start - 30), (width - 40, hud_y_start + 140), (45, 40, 35), 1)

            lbl_text, lbl_color = model_names[st.session_state.active_model]
            cv2.putText(frame, "ACTIVE MODEL:", (hud_x, hud_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 150, 140), 1, cv2.LINE_AA)
            cv2.putText(frame, lbl_text, (hud_x, hud_y_start + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, lbl_color, 2, cv2.LINE_AA)

            gesture_lbl = "ASSEMBLED"
            gesture_col = (160, 214, 6)
            if st.session_state.gesture_state == 'OPEN':
                gesture_lbl = "SCATTERED"
                gesture_col = (126, 71, 255)
            elif st.session_state.gesture_state == 'NONE':
                gesture_lbl = "WAITING..."
                gesture_col = (160, 150, 140)

            cv2.putText(frame, "GESTURE STATE:", (hud_x, hud_y_start + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 150, 140), 1, cv2.LINE_AA)
            cv2.putText(frame, gesture_lbl, (hud_x, hud_y_start + 77), cv2.FONT_HERSHEY_SIMPLEX, 0.58, gesture_col, 2, cv2.LINE_AA)

            status_text = "HAND TRACKED" if is_tracking else "SEARCHING FOR HAND..."
            status_col = (6, 214, 160) if is_tracking else (140, 150, 160)
            cv2.putText(frame, status_text, (hud_x, hud_y_start + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_col, 1, cv2.LINE_AA)

            second_hand_text = (
                f"2ND HAND: ZOOM {zoom_scale:.2f}x" if st.session_state.second_hand_tracking
                else "2ND HAND: NOT DETECTED"
            )
            second_hand_col = (255, 191, 0) if st.session_state.second_hand_tracking else (110, 105, 100)
            cv2.putText(frame, second_hand_text, (hud_x, hud_y_start + 123), cv2.FONT_HERSHEY_SIMPLEX, 0.40, second_hand_col, 1, cv2.LINE_AA)

            curr_time = time.time()
            fps = int(1.0 / max(0.001, curr_time - prev_time))
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (6, 214, 160), 2, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            with status_placeholder.container():
                st.write(f"**Model:** {lbl_text.title()}")
                st.write(f"**Gesture:** {gesture_lbl}")
                if st.session_state.second_hand_tracking:
                    st.write(f"**Zoom:** {zoom_scale:.2f}x")
                else:
                    st.write("**Zoom:** (show second hand)")
                st.write(f"**FPS:** {fps}")

        cap.release()
        hands.close()
else:
    frame_placeholder.info("Camera is stopped. Check **Start camera** in the sidebar to begin.")