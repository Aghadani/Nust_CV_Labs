"""
Lab 08 - Setup and Verification Utility
========================================

This script helps you:
1. Verify all required packages are installed
2. Check your folder structure
3. Create sample folders
4. Test basic functionality

Run this BEFORE starting the lab tasks!

Author: Computer Vision Lab
"""

import sys
import os

def verify_packages():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("STEP 1: Verifying Required Packages")
    print("=" * 60)
    print()
    
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'pickle': 'built-in'
    }
    
    all_installed = True
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {module:<15} INSTALLED")
        except ImportError:
            print(f"✗ {module:<15} MISSING - Install with: pip install {package}")
            all_installed = False
    
    print()
    
    if all_installed:
        print("SUCCESS: All packages are installed!")
        
        # Check SIFT availability
        try:
            import cv2
            sift = cv2.SIFT_create()
            print("✓ SIFT is available in OpenCV")
        except Exception as e:
            print(f"✗ SIFT is NOT available: {e}")
            print("  Install opencv-contrib-python:")
            print("  pip install opencv-contrib-python")
            all_installed = False
    else:
        print("ERROR: Some packages are missing!")
        print("\nQuick install command:")
        print("pip install opencv-python opencv-contrib-python numpy scikit-learn matplotlib seaborn scipy")
    
    print()
    return all_installed

def create_folder_structure():
    """Create the recommended folder structure."""
    print("=" * 60)
    print("STEP 2: Creating Folder Structure")
    print("=" * 60)
    print()
    
    folders = [
        'training_images',
        'test_images',
        'dataset/cars',
        'dataset/flowers'
    ]
    
    print("Creating folders:")
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"✓ Created: {folder}")
        except Exception as e:
            print(f"✗ Failed to create {folder}: {e}")
    
    print()
    print("Folder structure created!")
    print("\nNow add your images to these folders:")
    print("  training_images/  - 20+ images for vocabulary building")
    print("  test_images/      - 2-3 images for testing histograms")
    print("  dataset/cars/     - Car images for classification")
    print("  dataset/flowers/  - Flower images for classification")
    print()

def check_folder_contents():
    """Check if folders contain images."""
    print("=" * 60)
    print("STEP 3: Checking Folder Contents")
    print("=" * 60)
    print()
    
    folders_to_check = [
        'training_images',
        'test_images',
        'dataset/cars',
        'dataset/flowers'
    ]
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for folder in folders_to_check:
        if not os.path.exists(folder):
            print(f"✗ {folder:<25} DOES NOT EXIST")
            continue
        
        files = os.listdir(folder)
        image_files = [f for f in files 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        count = len(image_files)
        
        if count == 0:
            print(f"✗ {folder:<25} EMPTY (0 images)")
        elif count < 10:
            print(f"⚠ {folder:<25} {count} images (Need at least 10)")
        else:
            print(f"✓ {folder:<25} {count} images")
    
    print()

def test_sift():
    """Test SIFT feature extraction on a sample image."""
    print("=" * 60)
    print("STEP 4: Testing SIFT Feature Extraction")
    print("=" * 60)
    print()
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        print("Creating test image...")
        test_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Add some features
        cv2.circle(test_img, (100, 100), 50, (255, 255, 255), -1)
        cv2.rectangle(test_img, (200, 200), (300, 300), (255, 255, 255), -1)
        cv2.line(test_img, (50, 300), (350, 300), (255, 255, 255), 5)
        
        # Convert to grayscale
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # Extract SIFT features
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        print(f"✓ SIFT extraction successful!")
        print(f"  Found {len(keypoints)} keypoints")
        print(f"  Descriptor shape: {descriptors.shape}")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ SIFT test failed: {e}")
        print()
        return False

def test_kmeans():
    """Test K-Means clustering."""
    print("=" * 60)
    print("STEP 5: Testing K-Means Clustering")
    print("=" * 60)
    print()
    
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        
        # Create sample data
        print("Creating sample data...")
        data = np.random.randn(1000, 128)  # Simulate SIFT descriptors
        
        # Apply K-Means
        print("Applying K-Means with K=50...")
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
        kmeans.fit(data)
        
        print(f"✓ K-Means clustering successful!")
        print(f"  Cluster centers shape: {kmeans.cluster_centers_.shape}")
        print(f"  Number of iterations: {kmeans.n_iter_}")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ K-Means test failed: {e}")
        print()
        return False

def display_summary():
    """Display summary and next steps."""
    print("=" * 60)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("-----------")
    print("1. Add images to the created folders")
    print("2. Modify the CONFIGURATION section in each task script")
    print("3. Run tasks in order:")
    print("   a. python task1_build_vocabulary.py")
    print("   b. python task2_image_representation.py")
    print("   c. python task3_classification.py")
    print()
    print("Need help? Check STUDENT_GUIDE.txt for detailed instructions!")
    print()

def main():
    """Run all verification steps."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "LAB 08 - SETUP VERIFICATION UTILITY" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Step 1: Verify packages
    packages_ok = verify_packages()
    
    if not packages_ok:
        print("Please install missing packages before continuing!")
        print("Run: pip install opencv-python opencv-contrib-python numpy scikit-learn matplotlib seaborn scipy")
        return
    
    # Step 2: Create folders
    create_folder_structure()
    
    # Step 3: Check folder contents
    check_folder_contents()
    
    # Step 4: Test SIFT
    sift_ok = test_sift()
    
    # Step 5: Test K-Means
    kmeans_ok = test_kmeans()
    
    # Summary
    display_summary()
    
    # Final status
    if packages_ok and sift_ok and kmeans_ok:
        print("✓ SYSTEM READY - You can start the lab tasks!")
    else:
        print("⚠ ISSUES DETECTED - Please fix the errors above")
    
    print()

if __name__ == "__main__":
    main()
