# Computer Vision Lab 09: 2D Geometric Transformations

This repository contains the implementation of **Lab 09: 2D Transformations Applied to Images**. The project explores the mathematical foundations of image manipulation, including transformation matrices, homogeneous coordinates, and interpolation methods using Python and OpenCV.

## 📌 Overview
Geometric transformations are essential for tasks such as image registration, data augmentation for machine learning, and document processing. This lab implements basic to advanced 2D transformations and demonstrates practical applications like document skew correction.

---

## 📂 Project Structure

### 1. Basic Transformations (`lab09_q1_basic_transformations.py`)
Focuses on the core affine transformations:
* **Translation:** Shifting an image by $(t_x, t_y)$ offsets using a $2 \times 3$ matrix.
* **Rotation:** Rotating the image around its center point using `cv2.getRotationMatrix2D`.
* **Scaling:** Resizing the image using specified scale factors ($s_x, s_y$).
* **Output:** Generates a $2 \times 2$ visualization grid showcasing the original, translated, rotated, and scaled images.

### 2. Advanced & Composite Transformations (`lab09_q2_advanced_transformations.py`)
Explores complex manipulations and matrix operations:
* **Shearing:** Slanting the image along the x-axis (factor 0.5) and y-axis (factor 0.3).
* **Reflection:** Mirroring the image horizontally and vertically using `cv2.flip`.
* **Composite Transformation:** Combining rotation (30°), scaling (0.8x), and translation (100, 50) into a single unified operation using $3 \times 3$ homogeneous matrix multiplication.
* **Output:** Generates a $2 \times 3$ visualization grid with labels for each operation.

### 3. Interpolation & Practical Application (`lab09_q3_interpolation_practical.py`)
Analyzes image quality and real-world problem-solving:
* **Interpolation Comparison:** A side-by-side analysis of **Nearest Neighbor**, **Bilinear**, and **Bicubic** methods during a $60^\circ$ rotation.
* **Skew Correction:** 1.  Simulates a "skewed" document using shear transformations.
    2.  Calculates and applies the **inverse transformation matrix** to restore the document to its original upright position.
* **Output:** Comparison grids for interpolation methods and a "Before vs. After" view of the document correction.

---

## 🛠 Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install opencv-python numpy matplotlib
```
## 🚀 How to Run
1. Place your target image in the project directory.
2. Ensure the file names in the scripts match your image file name.
3. Run the scripts sequentially:Bashpython lab09_q1_basic_transformations.py
```bash
python lab09_q1_basic_transformations.py
python lab09_q2_advanced_transformations.py
python lab09_q3_interpolation_practical.py
```
## 🧠 Key Learning Points
* **Matrix Representation:** All transformations are represented as matrices. For translation, $3 \times 3$ homogeneous coordinates are required to combine it with rotation and scaling.
* **Interpolation Trade-offs:** Nearest Neighbor: Fast but causes blocky "aliasing" effects.Bilinear: Smooth results with moderate computation.Bicubic: Highest visual quality and sharpest edges, but computationally more expensive.
* **Inverse Transformations:** To "undo" a geometric distortion, we calculate the inverse of the transformation matrix ($M^{-1}$).
---
Course: CS-474 Computer VisionLab: 09 - 2D Transformations
Semester: 6th Semester
