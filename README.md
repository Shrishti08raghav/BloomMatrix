# 🌸 Bloom Matrix: Interactive 3D Holographic Flower Showcase

**Bloom Matrix** is a real-time, interactive 3D holographic flower showcase built using **Python**, **OpenCV**, and **MediaPipe**. By utilizing computer vision, this project allows users to manipulate and morph procedurally generated 3D particle flowers using natural hand gestures.

---

## ✨ Features
* 🌿 **5 Procedural 3D Flowers**: Beautiful mathematical particle designs for a Velvet Red Rose, Sacred Pink Lotus, Golden Sunflower, Sunset Orange Tulip, and Stargazer Pink Lily.
* 🎮 **Independent Dual-Hand Controls**:
  * **Left Hand (Flower)**: Move to tilt/rotate on X/Y axes, open hand to scatter particles, close hand to morph into the next model.
  * **Right Hand (Box)**: Move to spin the wireframe 360° via a virtual trackball, open hand to scale the box.
* 🔒 **In-Place Freezing**: Lowering a hand locks that element's orientation in place while maintaining its smooth automatic background spin.
* 🌈 **Supercharge Mode**: Showing both hands simultaneously triggers a flashing neon-rainbow hyper-spin.
* 🚀 **Z-Depth Sorting**: Implementation of the Painter's Algorithm in NumPy for accurate overlapping 3D particle occlusion.

---

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/bloom-matrix.git
   cd bloom-matrix
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install opencv-python numpy mediapipe
   ```

---

## 🚀 How to Run

Run the application using:
```bash
python app.py
```

* **Controls**:
  * Put up your **Left Hand** to control the flower.
  * Put up your **Right Hand** to control the bounding box.
  * Lower either hand to **freeze** its tilt angle.
  * Put up **both hands** to engage the flashing rainbow hyper-spin!
  * Press `q` or `ESC` on the camera window to exit.
