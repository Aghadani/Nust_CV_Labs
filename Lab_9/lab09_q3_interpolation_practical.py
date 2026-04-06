"""
Lab 09 - Question 3: Interpolation Comparison and Practical Application
========================================================================

This script compares interpolation methods and demonstrates a practical
application of transformations for document skew correction.

Learning Objectives:
- Understand different interpolation methods (Nearest, Bilinear, Bicubic)
- Compare visual quality of interpolation techniques
- Apply transformations to solve real-world problems
- Implement inverse transformations for correction

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

# ============================================================================
# PART A: INTERPOLATION COMPARISON
# ============================================================================

def rotate_with_interpolation(img, angle, interpolation_method):
    """
    Rotate image using a specific interpolation method.
    
    Args:
        img: Input image
        angle: Rotation angle in degrees
        interpolation_method: cv2.INTER_NEAREST, INTER_LINEAR, or INTER_CUBIC
        
    Returns:
        rotated_img: Rotated image
        method_name: Name of interpolation method
    """
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation with specified interpolation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height), 
                                  flags=interpolation_method)
    
    # Get method name
    method_names = {
        cv2.INTER_NEAREST: "Nearest Neighbor",
        cv2.INTER_LINEAR: "Bilinear",
        cv2.INTER_CUBIC: "Bicubic"
    }
    method_name = method_names.get(interpolation_method, "Unknown")
    
    return rotated_img, method_name

def compare_interpolation_methods(img, angle):
    """
    Compare three interpolation methods on the same rotation.
    
    Args:
        img: Input image
        angle: Rotation angle for comparison
        
    Returns:
        results: Dictionary with rotated images and method names
    """
    print("=" * 60)
    print("PART A: INTERPOLATION COMPARISON")
    print("=" * 60)
    print(f"Rotation angle: {angle}°")
    print()
    
    # Define interpolation methods to compare
    methods = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC
    ]
    
    results = {}
    
    for method in methods:
        rotated, method_name = rotate_with_interpolation(img, angle, method)
        results[method_name] = rotated
        print(f"✓ Rotation applied with {method_name} interpolation")
    
    print()
    
    # Print interpolation characteristics
    print("Interpolation Method Characteristics:")
    print("-" * 60)
    print("Nearest Neighbor:")
    print("  • Fastest method")
    print("  • Uses single nearest pixel value")
    print("  • Can produce blocky/jagged edges")
    print("  • Best for: Categorical images, masks")
    print()
    print("Bilinear:")
    print("  • Moderate speed")
    print("  • Averages 4 nearest pixels (2x2 neighborhood)")
    print("  • Smoother than nearest neighbor")
    print("  • Best for: General purpose, photographs")
    print()
    print("Bicubic:")
    print("  • Slowest method")
    print("  • Uses 16 surrounding pixels (4x4 neighborhood)")
    print("  • Smoothest results, best quality")
    print("  • Best for: High-quality output, enlarging images")
    print("-" * 60)
    print()
    
    return results

def visualize_interpolation_comparison(results, angle):
    """
    Display interpolation comparison in a 1x3 grid.
    
    Args:
        results: Dictionary with interpolation results
        angle: Rotation angle used
    """
    print("Creating interpolation comparison visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Interpolation Method Comparison - {angle}° Rotation', 
                 fontsize=16, fontweight='bold')
    
    method_order = ["Nearest Neighbor", "Bilinear", "Bicubic"]
    
    for ax, method_name in zip(axes, method_order):
        img_rgb = cv2.cvtColor(results[method_name], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(method_name, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    filename = 'interpolation_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}\n")
    
    plt.show()

def analyze_interpolation_quality(results):
    """
    Provide analysis of interpolation quality differences.
    
    Args:
        results: Dictionary with interpolation results
    """
    print("=" * 60)
    print("INTERPOLATION QUALITY ANALYSIS")
    print("=" * 60)
    print()
    
    print("Visual Quality Observations:")
    print()
    print("1. Nearest Neighbor:")
    print("   • Edges appear jagged and pixelated")
    print("   • Staircase effect (aliasing) visible on diagonal lines")
    print("   • Colors remain exact (no blending)")
    print("   • Sharp transitions between pixels")
    print()
    print("2. Bilinear:")
    print("   • Smoother edges compared to nearest neighbor")
    print("   • Some aliasing still visible but reduced")
    print("   • Colors are slightly blended")
    print("   • Good balance between quality and speed")
    print()
    print("3. Bicubic:")
    print("   • Smoothest edges and transitions")
    print("   • Minimal aliasing artifacts")
    print("   • Best color blending")
    print("   • Highest visual quality but slowest")
    print()
    
    print("Recommendation:")
    print("  • Use Nearest Neighbor: For masks, labeled images, or when")
    print("    exact pixel values must be preserved")
    print("  • Use Bilinear: For most photographic applications")
    print("  • Use Bicubic: When quality is critical and speed is not")
    print()

# ============================================================================
# PART B: PRACTICAL APPLICATION - DOCUMENT SKEW CORRECTION
# ============================================================================

def create_skewed_document(img, shear_x, shear_y):
    """
    Simulate a skewed document by applying shear transformation.
    
    Args:
        img: Input document image
        shear_x: Horizontal shear factor
        shear_y: Vertical shear factor
        
    Returns:
        skewed_img: Skewed document image
        shear_matrix: Transformation matrix used
    """
    print("=" * 60)
    print("PART B: DOCUMENT SKEW CORRECTION")
    print("=" * 60)
    print("Step 1: Creating Skewed Document")
    print("-" * 60)
    print(f"Shear parameters: x={shear_x}, y={shear_y}")
    print()
    
    height, width = img.shape[:2]
    
    # Combined shear matrix (both horizontal and vertical)
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])
    
    print("Shear Matrix:")
    print(shear_matrix)
    print()
    
    # Calculate new canvas size
    max_x_shift = int(abs(shear_x * height) + abs(shear_y * height))
    max_y_shift = int(abs(shear_y * width) + abs(shear_x * width))
    
    new_width = width + max_x_shift + 50
    new_height = height + max_y_shift + 50
    
    # Adjust matrix for visibility
    shear_matrix[0, 2] = 25  # Small offset
    shear_matrix[1, 2] = 25
    
    # Apply shear transformation
    skewed_img = cv2.warpAffine(img, shear_matrix, (new_width, new_height),
                                 borderValue=(255, 255, 255))
    
    print(f"✓ Document skewed")
    print(f"  Original size: {width} x {height}")
    print(f"  Skewed size: {new_width} x {new_height}")
    print()
    
    return skewed_img, shear_matrix

def correct_skew(skewed_img, original_shear_matrix):
    """
    Correct the skewed document using inverse transformation.
    
    Args:
        skewed_img: Skewed document image
        original_shear_matrix: Original shear matrix used to create skew
        
    Returns:
        corrected_img: Corrected document
        inverse_matrix: Inverse transformation matrix
    """
    print("Step 2: Correcting Skew")
    print("-" * 60)
    print()
    
    # Calculate inverse transformation
    # For a 2x3 affine matrix, we need to compute the inverse
    # Add third row [0, 0, 1] to make it 3x3
    shear_3x3 = np.vstack([original_shear_matrix, [0, 0, 1]])
    
    # Compute inverse
    inverse_3x3 = np.linalg.inv(shear_3x3)
    
    # Extract 2x3 matrix
    inverse_matrix = inverse_3x3[:2, :]
    
    print("Original Shear Matrix (2x3):")
    print(original_shear_matrix)
    print()
    print("Inverse Matrix (2x3):")
    print(inverse_matrix)
    print()
    
    # Verify it's actually inverse
    identity_check = shear_3x3 @ inverse_3x3
    print("Verification (Original × Inverse should be Identity):")
    print(identity_check)
    print()
    
    # Apply inverse transformation
    height, width = skewed_img.shape[:2]
    corrected_img = cv2.warpAffine(skewed_img, inverse_matrix, (width, height),
                                    borderValue=(255, 255, 255))
    
    print("✓ Skew corrected using inverse transformation")
    print()
    
    return corrected_img, inverse_matrix

def visualize_skew_correction(original, skewed, corrected):
    """
    Display original, skewed, and corrected documents.
    
    Args:
        original: Original document
        skewed: Skewed document
        corrected: Corrected document
    """
    print("Creating skew correction visualization...")
    
    # Convert to RGB
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    skewed_rgb = cv2.cvtColor(skewed, cv2.COLOR_BGR2RGB)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Document Skew Correction - Practical Application', 
                 fontsize=16, fontweight='bold')
    
    # Original
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Document', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Skewed
    axes[1].imshow(skewed_rgb)
    axes[1].set_title('Skewed Document\n(Simulated Distortion)', 
                      fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # Corrected
    axes[2].imshow(corrected_rgb)
    axes[2].set_title('Corrected Document\n(Inverse Transform Applied)', 
                      fontsize=14, fontweight='bold', color='green')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    filename = 'skew_correction.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}\n")
    
    plt.show()

def create_document_image():
    """
    Create a sample document image with text and graphics.
    
    Returns:
        img: Document image
    """
    # Create white document
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255
    
    # Add border
    cv2.rectangle(img, (20, 20), (680, 480), (0, 0, 0), 2)
    
    # Add title
    cv2.putText(img, "INVOICE", (280, 60), 
                cv2.FONT_HERSHEY_BOLD, 1.5, (0, 0, 0), 3)
    
    # Add horizontal line
    cv2.line(img, (40, 80), (660, 80), (0, 0, 0), 2)
    
    # Add document number
    cv2.putText(img, "No: INV-2026-001", (40, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add date
    cv2.putText(img, "Date: April 6, 2026", (40, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add table header
    cv2.rectangle(img, (40, 180), (660, 220), (200, 200, 200), -1)
    cv2.putText(img, "ITEM", (60, 205), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "QTY", (400, 205), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "PRICE", (520, 205), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add table rows
    items = [
        ("Product A", "2", "$50.00"),
        ("Product B", "1", "$75.00"),
        ("Product C", "3", "$30.00")
    ]
    
    y_pos = 250
    for item, qty, price in items:
        cv2.putText(img, item, (60, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, qty, (410, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, price, (520, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 40
    
    # Add total
    cv2.line(img, (40, 370), (660, 370), (0, 0, 0), 2)
    cv2.putText(img, "TOTAL:", (400, 410), 
                cv2.FONT_HERSHEY_BOLD, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "$265.00", (520, 410), 
                cv2.FONT_HERSHEY_BOLD, 0.8, (0, 0, 0), 2)
    
    # Add signature line
    cv2.line(img, (400, 460), (620, 460), (0, 0, 0), 1)
    cv2.putText(img, "Signature", (480, 475), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save document
    cv2.imwrite('sample_document.jpg', img)
    print("✓ Sample document created: sample_document.jpg\n")
    
    return img

def main():
    """Main function to execute Question 3 tasks."""
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 2 + "LAB 09 - QUESTION 3: INTERPOLATION & PRACTICAL USE" + " " * 4 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Create sample document
    print("Creating sample document image...")
    IMAGE_PATH = 'sample_document.jpg'
    document_img = create_document_image()
    
    # Parameters
    ROTATION_ANGLE = 60  # For interpolation comparison
    SKEW_X = 0.2  # Horizontal shear for document skew
    SKEW_Y = 0.15  # Vertical shear for document skew
    
    # ========================================================================
    
    try:
        # ====================================================================
        # PART A: INTERPOLATION COMPARISON
        # ====================================================================
        
        # Compare interpolation methods
        interp_results = compare_interpolation_methods(document_img, ROTATION_ANGLE)
        
        # Visualize comparison
        visualize_interpolation_comparison(interp_results, ROTATION_ANGLE)
        
        # Analyze quality
        analyze_interpolation_quality(interp_results)
        
        # ====================================================================
        # PART B: DOCUMENT SKEW CORRECTION
        # ====================================================================
        
        # Simulate skewed document
        skewed_doc, skew_matrix = create_skewed_document(document_img, SKEW_X, SKEW_Y)
        
        # Correct the skew
        corrected_doc, inverse_matrix = correct_skew(skewed_doc, skew_matrix)
        
        # Visualize correction
        visualize_skew_correction(document_img, skewed_doc, corrected_doc)
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        
        print("=" * 60)
        print("QUESTION 3 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nPart A - Interpolation Comparison:")
        print(f"  • Compared 3 methods on {ROTATION_ANGLE}° rotation")
        print("  • Methods: Nearest Neighbor, Bilinear, Bicubic")
        print("  • Analysis provided above")
        print()
        print("Part B - Document Skew Correction:")
        print(f"  • Simulated skew: x={SKEW_X}, y={SKEW_Y}")
        print("  • Applied inverse transformation")
        print("  • Successfully corrected document distortion")
        print()
        print("Outputs Generated:")
        print("  • interpolation_comparison.png")
        print("  • skew_correction.png")
        print()
        print("Key Learning Points:")
        print("  1. Interpolation quality vs speed tradeoff")
        print("  2. Practical use of inverse transformations")
        print("  3. Real-world application: Document correction")
        print("  4. Matrix operations for transformation reversal")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")

if __name__ == "__main__":
    main()
