"""
Lab 09 - Question 2: Advanced Transformations
==============================================

This script implements Shearing, Reflection, and Composite transformations on images.

Learning Objectives:
- Understand and implement shearing transformations
- Apply reflection (flipping) operations
- Create composite transformations by matrix multiplication
- Visualize advanced transformation effects

Author: Computer Vision Lab
Course: CS-474
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from file."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    print(f"Image loaded: {img.shape[1]} x {img.shape[0]} pixels\n")
    return img

def apply_horizontal_shearing(img, shear_factor):
    """
    Apply horizontal shearing transformation.
    
    Horizontal shearing slants the image along the x-axis.
    Each point (x, y) is mapped to (x + shear_factor * y, y).
    
    Args:
        img: Input image
        shear_factor: Shearing factor (0.5 = moderate slant)
        
    Returns:
        sheared_img: Transformed image
        shear_matrix: 2x3 transformation matrix
    """
    print("=" * 60)
    print("HORIZONTAL SHEARING")
    print("=" * 60)
    print(f"Shear factor: {shear_factor}")
    print()
    
    height, width = img.shape[:2]
    
    # Horizontal shear matrix:
    # [1   shear_factor   0]
    # [0   1              0]
    shear_matrix = np.float32([
        [1, shear_factor, 0],
        [0, 1, 0]
    ])
    
    print("Horizontal Shear Matrix:")
    print(shear_matrix)
    print()
    print("Effect: x_new = x + shear_factor * y")
    print("        y_new = y")
    print()
    
    # Calculate new width to accommodate the sheared image
    # The maximum shift occurs at the top or bottom of the image
    max_shift = int(abs(shear_factor * height))
    new_width = width + max_shift
    
    # Adjust the transformation matrix to keep image visible
    if shear_factor > 0:
        # Positive shear - shift happens at top
        shear_matrix[0, 2] = 0
    else:
        # Negative shear - shift happens at bottom
        shear_matrix[0, 2] = abs(max_shift)
    
    sheared_img = cv2.warpAffine(img, shear_matrix, (new_width, height))
    
    print(f"✓ Horizontal shearing applied")
    print(f"  New width: {new_width} (original: {width})")
    print()
    
    return sheared_img, shear_matrix

def apply_vertical_shearing(img, shear_factor):
    """
    Apply vertical shearing transformation.
    
    Vertical shearing slants the image along the y-axis.
    Each point (x, y) is mapped to (x, y + shear_factor * x).
    
    Args:
        img: Input image
        shear_factor: Shearing factor
        
    Returns:
        sheared_img: Transformed image
        shear_matrix: 2x3 transformation matrix
    """
    print("=" * 60)
    print("VERTICAL SHEARING")
    print("=" * 60)
    print(f"Shear factor: {shear_factor}")
    print()
    
    height, width = img.shape[:2]
    
    # Vertical shear matrix:
    # [1   0   0]
    # [shear_factor   1   0]
    shear_matrix = np.float32([
        [1, 0, 0],
        [shear_factor, 1, 0]
    ])
    
    print("Vertical Shear Matrix:")
    print(shear_matrix)
    print()
    print("Effect: x_new = x")
    print("        y_new = y + shear_factor * x")
    print()
    
    # Calculate new height to accommodate the sheared image
    max_shift = int(abs(shear_factor * width))
    new_height = height + max_shift
    
    # Adjust for visibility
    if shear_factor > 0:
        shear_matrix[1, 2] = 0
    else:
        shear_matrix[1, 2] = abs(max_shift)
    
    sheared_img = cv2.warpAffine(img, shear_matrix, (width, new_height))
    
    print(f"✓ Vertical shearing applied")
    print(f"  New height: {new_height} (original: {height})")
    print()
    
    return sheared_img, shear_matrix

def apply_horizontal_reflection(img):
    """
    Apply horizontal reflection (flip left-right).
    
    Creates a mirror image across the vertical axis.
    
    Args:
        img: Input image
        
    Returns:
        reflected_img: Transformed image
    """
    print("=" * 60)
    print("HORIZONTAL REFLECTION (Flip Left-Right)")
    print("=" * 60)
    print()
    
    # OpenCV provides a simple flip function
    # flipCode = 1 means flip horizontally
    reflected_img = cv2.flip(img, 1)
    
    print("Reflection Matrix (conceptual):")
    print("[-1   0   width]")
    print("[ 0   1   0    ]")
    print()
    print("Effect: Mirrors image across vertical axis")
    print("✓ Horizontal reflection applied")
    print()
    
    return reflected_img

def apply_vertical_reflection(img):
    """
    Apply vertical reflection (flip top-bottom).
    
    Creates a mirror image across the horizontal axis.
    
    Args:
        img: Input image
        
    Returns:
        reflected_img: Transformed image
    """
    print("=" * 60)
    print("VERTICAL REFLECTION (Flip Top-Bottom)")
    print("=" * 60)
    print()
    
    # flipCode = 0 means flip vertically
    reflected_img = cv2.flip(img, 0)
    
    print("Reflection Matrix (conceptual):")
    print("[ 1   0   0     ]")
    print("[ 0  -1   height]")
    print()
    print("Effect: Mirrors image across horizontal axis")
    print("✓ Vertical reflection applied")
    print()
    
    return reflected_img

def create_composite_transformation(img, rotation_angle, scale_factor, tx, ty):
    """
    Create and apply a composite transformation.
    
    Combines multiple transformations: rotation + scaling + translation.
    The order of operations matters!
    
    Args:
        img: Input image
        rotation_angle: Rotation angle in degrees
        scale_factor: Scaling factor
        tx, ty: Translation offsets
        
    Returns:
        transformed_img: Result of composite transformation
        composite_matrix: Combined transformation matrix
    """
    print("=" * 60)
    print("COMPOSITE TRANSFORMATION")
    print("=" * 60)
    print(f"Operations: Rotation({rotation_angle}°) + Scale({scale_factor}x) + Translation({tx}, {ty})")
    print()
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Step 1: Create rotation matrix
    # getRotationMatrix2D automatically handles rotation around center
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale=1.0)
    
    print("Step 1 - Rotation Matrix:")
    print(rotation_matrix)
    print()
    
    # Step 2: Create scaling matrix
    scaling_matrix = np.float32([
        [scale_factor, 0, 0],
        [0, scale_factor, 0]
    ])
    
    print("Step 2 - Scaling Matrix:")
    print(scaling_matrix)
    print()
    
    # Step 3: Combine rotation and scaling
    # Convert rotation matrix to 3x3 for matrix multiplication
    rotation_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])
    scaling_3x3 = np.vstack([scaling_matrix, [0, 0, 1]])
    
    # Multiply matrices: First scale, then rotate
    intermediate = rotation_3x3 @ scaling_3x3
    
    # Step 4: Add translation
    intermediate[0, 2] += tx
    intermediate[1, 2] += ty
    
    # Extract 2x3 matrix for warpAffine
    composite_matrix = intermediate[:2, :]
    
    print("Step 3 - Composite Matrix (Rotation * Scaling + Translation):")
    print(composite_matrix)
    print()
    
    # Apply composite transformation
    # Calculate new canvas size to fit transformed image
    new_width = int(width * scale_factor) + abs(tx) + 100
    new_height = int(height * scale_factor) + abs(ty) + 100
    
    transformed_img = cv2.warpAffine(img, composite_matrix, (new_width, new_height))
    
    print("✓ Composite transformation applied")
    print(f"  Final size: {new_width} x {new_height}")
    print()
    
    return transformed_img, composite_matrix

def resize_for_display(img, target_height):
    """Resize image to target height while maintaining aspect ratio."""
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(img, (new_width, target_height))

def visualize_all_transformations(original, h_shear, v_shear, h_reflect, v_reflect, composite):
    """
    Display all transformations in a 2x3 grid.
    
    Args:
        original: Original image
        h_shear: Horizontally sheared image
        v_shear: Vertically sheared image
        h_reflect: Horizontally reflected image
        v_reflect: Vertically reflected image
        composite: Composite transformed image
    """
    print("=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    print()
    
    # Resize all images to same height for consistent display
    target_height = 300
    
    original_display = resize_for_display(original, target_height)
    h_shear_display = resize_for_display(h_shear, target_height)
    v_shear_display = resize_for_display(v_shear, target_height)
    h_reflect_display = resize_for_display(h_reflect, target_height)
    v_reflect_display = resize_for_display(v_reflect, target_height)
    composite_display = resize_for_display(composite, target_height)
    
    # Convert to RGB
    images_rgb = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        for img in [original_display, h_shear_display, v_shear_display,
                    h_reflect_display, v_reflect_display, composite_display]
    ]
    
    # Create 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('2D Transformations - Advanced Operations', 
                 fontsize=18, fontweight='bold')
    
    titles = [
        'Original Image',
        'Horizontal Shearing\n(factor = 0.5)',
        'Vertical Shearing\n(factor = 0.3)',
        'Horizontal Reflection\n(Flip Left-Right)',
        'Vertical Reflection\n(Flip Top-Bottom)',
        'Composite Transform\n(Rotate + Scale + Translate)'
    ]
    
    # Display images
    for idx, (ax, img, title) in enumerate(zip(axes.flat, images_rgb, titles)):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    filename = 'advanced_transformations_grid.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved as: {filename}")
    
    plt.show()

def create_sample_image():
    """Create a sample image with asymmetric features to show transformations clearly."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw asymmetric shapes to make transformations more visible
    # Arrow pointing right
    arrow = np.array([
        [100, 200], [200, 200], [200, 150], 
        [300, 250], [200, 350], [200, 300], [100, 300]
    ], np.int32)
    cv2.fillPoly(img, [arrow], (255, 100, 100))
    
    # Letter "F" shape
    cv2.rectangle(img, (400, 100), (450, 300), (100, 100, 255), -1)
    cv2.rectangle(img, (450, 100), (550, 150), (100, 100, 255), -1)
    cv2.rectangle(img, (450, 190), (520, 240), (100, 100, 255), -1)
    
    # Text
    cv2.putText(img, "Transform", (200, 380), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    cv2.imwrite('sample_image_advanced.jpg', img)
    print("✓ Sample image created: sample_image_advanced.jpg\n")
    
    return img

def main():
    """Main function to execute Question 2 tasks."""
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 6 + "LAB 09 - QUESTION 2: ADVANCED TRANSFORMATIONS" + " " * 7 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Create sample image
    print("Creating sample image...")
    IMAGE_PATH = 'sample_image_advanced.jpg'
    create_sample_image()
    
    # Transformation parameters
    H_SHEAR_FACTOR = 0.5
    V_SHEAR_FACTOR = 0.3
    COMPOSITE_ROTATION = 30
    COMPOSITE_SCALE = 0.8
    COMPOSITE_TX = 100
    COMPOSITE_TY = 50
    
    # ========================================================================
    
    try:
        # Load image
        img = load_image(IMAGE_PATH)
        
        # Task 1: Horizontal Shearing
        h_sheared, h_shear_matrix = apply_horizontal_shearing(img, H_SHEAR_FACTOR)
        
        # Task 2: Vertical Shearing
        v_sheared, v_shear_matrix = apply_vertical_shearing(img, V_SHEAR_FACTOR)
        
        # Task 3: Horizontal Reflection
        h_reflected = apply_horizontal_reflection(img)
        
        # Task 4: Vertical Reflection
        v_reflected = apply_vertical_reflection(img)
        
        # Task 5: Composite Transformation
        composite, composite_matrix = create_composite_transformation(
            img, COMPOSITE_ROTATION, COMPOSITE_SCALE, COMPOSITE_TX, COMPOSITE_TY
        )
        
        # Task 6: Visualize all transformations
        visualize_all_transformations(img, h_sheared, v_sheared, 
                                       h_reflected, v_reflected, composite)
        
        # Summary
        print("\n" + "=" * 60)
        print("QUESTION 2 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTransformations Applied:")
        print(f"  1. Horizontal Shearing: factor = {H_SHEAR_FACTOR}")
        print(f"  2. Vertical Shearing: factor = {V_SHEAR_FACTOR}")
        print(f"  3. Horizontal Reflection (flip)")
        print(f"  4. Vertical Reflection (flip)")
        print(f"  5. Composite: Rotate({COMPOSITE_ROTATION}°) + "
              f"Scale({COMPOSITE_SCALE}) + Translate({COMPOSITE_TX}, {COMPOSITE_TY})")
        print("\nOutputs Generated:")
        print("  • advanced_transformations_grid.png")
        print("  • All transformation matrices printed above")
        print()
        
        print("Key Observations:")
        print("  - Shearing creates slanted/skewed effects")
        print("  - Reflections create mirror images")
        print("  - Composite transformations combine multiple operations")
        print("  - Matrix multiplication order affects the final result")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")

if __name__ == "__main__":
    main()
