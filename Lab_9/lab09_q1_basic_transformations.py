"""
Lab 09 - Question 1: Basic Transformations
===========================================

This script implements Translation, Rotation, and Scaling transformations on images.

Learning Objectives:
- Understand translation matrices and image shifting
- Apply rotation transformations with proper center point
- Implement scaling with different interpolation methods
- Visualize multiple transformations in a grid layout

Author: Computer Vision Lab
Course: CS-474
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        img: Loaded image in BGR format
    """
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"Image loaded successfully!")
    print(f"  Shape: {img.shape}")
    print(f"  Size: {img.shape[1]} x {img.shape[0]} pixels")
    print()
    
    return img

def apply_translation(img, tx, ty):
    """
    Apply translation (shifting) transformation to an image.
    
    Translation moves every pixel by a constant offset (tx, ty).
    
    Args:
        img: Input image
        tx: Translation in x-direction (positive = right)
        ty: Translation in y-direction (positive = down)
        
    Returns:
        translated_img: Transformed image
        translation_matrix: 2x3 transformation matrix
    """
    print("=" * 60)
    print("TRANSLATION TRANSFORMATION")
    print("=" * 60)
    print(f"Parameters: tx={tx}, ty={ty}")
    print()
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create translation matrix
    # Translation matrix format:
    # [1   0   tx]
    # [0   1   ty]
    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
    
    print("Translation Matrix:")
    print(translation_matrix)
    print()
    
    # Apply translation using warpAffine
    translated_img = cv2.warpAffine(img, translation_matrix, (width, height))
    
    print(f"✓ Translation applied: shifted by ({tx}, {ty})")
    print()
    
    return translated_img, translation_matrix

def apply_rotation(img, angle_degrees, center=None):
    """
    Apply rotation transformation to an image.
    
    Rotation turns the image around a center point by a specified angle.
    Positive angles rotate counter-clockwise, negative rotate clockwise.
    
    Args:
        img: Input image
        angle_degrees: Rotation angle in degrees (negative = clockwise)
        center: Center point for rotation (default: image center)
        
    Returns:
        rotated_img: Transformed image
        rotation_matrix: 2x3 transformation matrix
    """
    print("=" * 60)
    print("ROTATION TRANSFORMATION")
    print("=" * 60)
    print(f"Parameters: angle={angle_degrees}°")
    print()
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Use image center as rotation center if not specified
    if center is None:
        center = (width // 2, height // 2)
    
    print(f"Rotation center: {center}")
    
    # Get rotation matrix using OpenCV
    # This creates a 2x3 matrix that combines rotation and translation
    # to keep the image centered
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    
    print("\nRotation Matrix:")
    print(rotation_matrix)
    print()
    
    # Manual calculation explanation
    angle_rad = np.radians(angle_degrees)
    print("Matrix components:")
    print(f"  cos({angle_degrees}°) = {np.cos(angle_rad):.4f}")
    print(f"  sin({angle_degrees}°) = {np.sin(angle_rad):.4f}")
    print()
    
    # Apply rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    print(f"✓ Rotation applied: {angle_degrees}° around {center}")
    print()
    
    return rotated_img, rotation_matrix

def apply_scaling(img, scale_x, scale_y, interpolation=cv2.INTER_LINEAR):
    """
    Apply scaling transformation to an image.
    
    Scaling changes the size of the image by multiplying coordinates
    by scale factors.
    
    Args:
        img: Input image
        scale_x: Scale factor for x-direction
        scale_y: Scale factor for y-direction
        interpolation: Interpolation method (INTER_NEAREST, INTER_LINEAR, INTER_CUBIC)
        
    Returns:
        scaled_img: Transformed image
        scaling_matrix: 2x3 transformation matrix
    """
    print("=" * 60)
    print("SCALING TRANSFORMATION")
    print("=" * 60)
    print(f"Parameters: scale_x={scale_x}, scale_y={scale_y}")
    
    # Get interpolation method name
    interp_names = {
        cv2.INTER_NEAREST: "Nearest Neighbor",
        cv2.INTER_LINEAR: "Bilinear",
        cv2.INTER_CUBIC: "Bicubic"
    }
    print(f"Interpolation: {interp_names.get(interpolation, 'Unknown')}")
    print()
    
    # Get original dimensions
    height, width = img.shape[:2]
    
    # Calculate new dimensions
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)
    
    print(f"Original size: {width} x {height}")
    print(f"New size: {new_width} x {new_height}")
    print()
    
    # Create scaling matrix
    # Scaling matrix format:
    # [sx  0   0]
    # [0   sy  0]
    scaling_matrix = np.float32([
        [scale_x, 0, 0],
        [0, scale_y, 0]
    ])
    
    print("Scaling Matrix:")
    print(scaling_matrix)
    print()
    
    # Apply scaling using resize (more straightforward than warpAffine for pure scaling)
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    
    print(f"✓ Scaling applied: {scale_x}x in width, {scale_y}x in height")
    print()
    
    return scaled_img, scaling_matrix

def visualize_transformations(original, translated, rotated, scaled):
    """
    Display original and transformed images in a 2x2 grid.
    
    Args:
        original: Original image
        translated: Translated image
        rotated: Rotated image
        scaled: Scaled image
    """
    print("=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    
    # Convert images from BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    translated_rgb = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    
    # Handle scaled image which might be different size
    # Resize it back to original size for display purposes
    h, w = original.shape[:2]
    scaled_display = cv2.resize(scaled, (w, h))
    scaled_rgb = cv2.cvtColor(scaled_display, cv2.COLOR_BGR2RGB)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('2D Transformations - Basic Operations', 
                 fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Translated image
    axes[0, 1].imshow(translated_rgb)
    axes[0, 1].set_title('Translation\n(50 right, 30 down)', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Rotated image
    axes[1, 0].imshow(rotated_rgb)
    axes[1, 0].set_title('Rotation\n(45° clockwise)', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Scaled image
    axes[1, 1].imshow(scaled_rgb)
    axes[1, 1].set_title('Scaling\n(1.5x both dimensions)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    filename = 'basic_transformations_grid.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved as: {filename}")
    
    plt.show()

def create_sample_image():
    """
    Create a sample image with geometric shapes for demonstration.
    
    Returns:
        img: Sample image
    """
    # Create a white canvas
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw some colored shapes
    # Blue rectangle
    cv2.rectangle(img, (100, 100), (250, 200), (255, 0, 0), -1)
    
    # Green circle
    cv2.circle(img, (400, 150), 60, (0, 255, 0), -1)
    
    # Red triangle
    triangle = np.array([[300, 250], [250, 350], [350, 350]], np.int32)
    cv2.fillPoly(img, [triangle], (0, 0, 255))
    
    # Add text
    cv2.putText(img, "2D Transform", (180, 380), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add border
    cv2.rectangle(img, (5, 5), (595, 395), (0, 0, 0), 3)
    
    # Save sample image
    cv2.imwrite('sample_image.jpg', img)
    print("✓ Sample image created: sample_image.jpg")
    
    return img

def main():
    """
    Main function to execute Question 1 tasks.
    """
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 8 + "LAB 09 - QUESTION 1: BASIC TRANSFORMATIONS" + " " * 9 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # ========================================================================
    # CONFIGURATION - Modify this section
    # ========================================================================
    
    # Option 1: Use your own image
    # IMAGE_PATH = 'path/to/your/image.jpg'
    
    # Option 2: Create a sample image
    print("Creating sample image for demonstration...")
    IMAGE_PATH = 'sample_image.jpg'
    create_sample_image()
    print()
    
    # Transformation parameters
    TX = 50  # Translation in x (right)
    TY = 30  # Translation in y (down)
    ROTATION_ANGLE = -45  # Negative for clockwise rotation
    SCALE_FACTOR = 1.5  # Scale factor for both dimensions
    
    # ========================================================================
    
    try:
        # Load image
        print("Loading image...")
        img = load_image(IMAGE_PATH)
        
        # Task 1: Apply Translation
        translated_img, trans_matrix = apply_translation(img, TX, TY)
        
        # Task 2: Apply Rotation
        rotated_img, rot_matrix = apply_rotation(img, ROTATION_ANGLE)
        
        # Task 3: Apply Scaling
        scaled_img, scale_matrix = apply_scaling(img, SCALE_FACTOR, SCALE_FACTOR, 
                                                  cv2.INTER_LINEAR)
        
        # Task 4: Visualize all transformations
        visualize_transformations(img, translated_img, rotated_img, scaled_img)
        
        # Summary
        print("\n" + "=" * 60)
        print("QUESTION 1 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTransformations Applied:")
        print(f"  1. Translation: ({TX}, {TY})")
        print(f"  2. Rotation: {ROTATION_ANGLE}°")
        print(f"  3. Scaling: {SCALE_FACTOR}x")
        print("\nOutputs Generated:")
        print("  • basic_transformations_grid.png")
        print("  • All transformation matrices printed above")
        print()
        
        # Additional insights
        print("Key Observations:")
        print("  - Translation shifts the entire image uniformly")
        print("  - Rotation may crop parts of the image at boundaries")
        print("  - Scaling changes the image resolution")
        print("  - Black regions appear where image data is missing")
        print()
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        print("\nPlease check:")
        print("  1. Image file exists and is readable")
        print("  2. OpenCV is properly installed")
        print("  3. Image path is correct")

if __name__ == "__main__":
    main()
