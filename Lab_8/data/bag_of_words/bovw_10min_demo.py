"""
=============================================================================
10-MINUTE BAG OF VISUAL WORDS (BoVW) DEMONSTRATION
=============================================================================

This is a simplified, complete example to demonstrate the BoVW concept
to students before they start their lab tasks.

Time: 10 minutes
Purpose: Show the entire pipeline from start to finish

INSTRUCTOR NOTES:
- Run this code while explaining each section
- Point out the 4 main steps of BoVW
- Show the outputs: vocabulary and histograms
- Emphasize that this is a simplified version
- Students will implement more detailed versions in their lab tasks

=============================================================================
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

# =============================================================================
# STEP 1: CREATE SAMPLE IMAGES (for demonstration)
# =============================================================================

def create_sample_images():
    """
    Create 6 simple sample images: 3 with circles, 3 with squares.
    In real lab, students will use actual image datasets.
    """
    print("=" * 70)
    print("STEP 1: Creating Sample Images")
    print("=" * 70)
    
    images = []
    
    # Create 3 images with CIRCLES (Category 1)
    for i in range(3):
        img = np.ones((200, 200), dtype=np.uint8) * 255  # White background
        # Draw circles at different positions
        cv2.circle(img, (70 + i*20, 100), 40, 0, -1)
        cv2.circle(img, (130, 80 + i*15), 30, 0, -1)
        images.append(img)
    
    # Create 3 images with SQUARES (Category 2)
    for i in range(3):
        img = np.ones((200, 200), dtype=np.uint8) * 255  # White background
        # Draw squares at different positions
        cv2.rectangle(img, (50 + i*10, 70), (110 + i*10, 130), 0, -1)
        cv2.rectangle(img, (120, 90 + i*10), (170, 140 + i*10), 0, -1)
        images.append(img)
    
    print(f"✓ Created {len(images)} sample images")
    print(f"  - 3 images with circles (Category 1)")
    print(f"  - 3 images with squares (Category 2)")
    print()
    
    return images

# =============================================================================
# STEP 2: EXTRACT SIFT FEATURES FROM ALL IMAGES
# =============================================================================

def extract_all_features(images):
    """
    Extract SIFT features from all images and collect them.
    This is the feature extraction step of BoVW.
    """
    print("=" * 70)
    print("STEP 2: Extracting SIFT Features")
    print("=" * 70)
    
    sift = cv2.SIFT_create()
    all_descriptors = []
    
    for i, img in enumerate(images):
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        if descriptors is not None:
            all_descriptors.append(descriptors)
            print(f"  Image {i+1}: {len(keypoints)} keypoints, {descriptors.shape} descriptors")
    
    # Combine all descriptors into one big array
    all_descriptors = np.vstack(all_descriptors)
    
    print()
    print(f"✓ Total descriptors collected: {all_descriptors.shape[0]}")
    print(f"  Each descriptor: {all_descriptors.shape[1]} dimensions (SIFT standard)")
    print()
    
    return all_descriptors

# =============================================================================
# STEP 3: CREATE VOCABULARY USING K-MEANS CLUSTERING
# =============================================================================

def create_vocabulary(all_descriptors, k=10):
    """
    Apply K-Means clustering to create visual vocabulary.
    The cluster centers become our "visual words".
    """
    print("=" * 70)
    print("STEP 3: Creating Visual Vocabulary (K-Means Clustering)")
    print("=" * 70)
    print(f"  Number of clusters (K): {k}")
    print(f"  This means we'll have {k} visual words in our vocabulary")
    print()
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)
    
    # The cluster centers are our visual vocabulary
    vocabulary = kmeans.cluster_centers_
    
    print(f"✓ Vocabulary created!")
    print(f"  Vocabulary shape: {vocabulary.shape}")
    print(f"  {vocabulary.shape[0]} visual words × {vocabulary.shape[1]} dimensions")
    print()
    
    return vocabulary

# =============================================================================
# STEP 4: CREATE BoVW HISTOGRAM FOR AN IMAGE
# =============================================================================

def create_bovw_histogram(image, vocabulary):
    """
    Convert a single image into a BoVW histogram.
    
    Process:
    1. Extract SIFT features from the image
    2. For each feature, find the nearest visual word
    3. Count how many times each visual word appears
    4. Normalize the histogram
    """
    print("  Converting image to BoVW histogram...")
    
    # Extract features from this image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None or len(descriptors) == 0:
        # No features found, return zero histogram
        return np.zeros(len(vocabulary))
    
    # For each descriptor, find the nearest visual word
    # This is called "vector quantization"
    from scipy.spatial.distance import cdist
    distances = cdist(descriptors, vocabulary, metric='euclidean')
    nearest_words = np.argmin(distances, axis=1)
    
    # Create histogram by counting occurrences
    histogram, _ = np.histogram(nearest_words, bins=len(vocabulary), 
                                range=(0, len(vocabulary)))
    
    # Normalize the histogram
    histogram = histogram.astype(float)
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    
    print(f"    Found {len(keypoints)} keypoints")
    print(f"    Created histogram with {len(histogram)} bins")
    print(f"    Histogram sum: {np.sum(histogram):.2f} (should be 1.0)")
    
    return histogram

# =============================================================================
# STEP 5: DEMONSTRATE THE COMPLETE PIPELINE
# =============================================================================

def demonstrate_bovw():
    """
    Complete demonstration of the BoVW pipeline.
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "BAG OF VISUAL WORDS - 10 MINUTE DEMO" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Step 1: Create sample images
    images = create_sample_images()
    
    # Step 2: Extract features from all images
    all_descriptors = extract_all_features(images)
    
    # Step 3: Create vocabulary (K=10 visual words)
    K = 10
    vocabulary = create_vocabulary(all_descriptors, k=K)
    
    # Save vocabulary (in real lab, students will load this later)
    with open('demo_vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    print("✓ Vocabulary saved as 'demo_vocabulary.pkl'")
    print()
    
    # Step 4: Convert images to BoVW histograms
    print("=" * 70)
    print("STEP 4: Creating BoVW Histograms")
    print("=" * 70)
    print()
    
    histograms = []
    labels = ['Circle'] * 3 + ['Square'] * 3
    
    for i, (img, label) in enumerate(zip(images, labels)):
        print(f"Image {i+1} ({label}):")
        hist = create_bovw_histogram(img, vocabulary)
        histograms.append(hist)
        print()
    
    # Step 5: Visualize the results
    print("=" * 70)
    print("STEP 5: Visualizing Results")
    print("=" * 70)
    print()
    
    visualize_results(images, histograms, labels, vocabulary)
    
    # Summary
    print("=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print()
    print("KEY POINTS TO REMEMBER:")
    print("  1. SIFT extracts local features from images")
    print("  2. K-Means clusters features into 'visual words'")
    print("  3. Each image becomes a histogram of visual word frequencies")
    print("  4. Similar images have similar histograms")
    print()
    print("IN YOUR LAB TASKS:")
    print("  • You'll use REAL images (not simple shapes)")
    print("  • You'll use larger K (50, 100, or more)")
    print("  • You'll train classifiers on these histograms")
    print("  • You'll evaluate with accuracy and confusion matrix")
    print()

# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================

def visualize_results(images, histograms, labels, vocabulary):
    """
    Create visualizations to show the BoVW process.
    """
    # Create figure with 2 parts: images and histograms
    fig = plt.figure(figsize=(16, 10))
    
    # Part 1: Show the original images
    print("Creating visualization...")
    for i in range(6):
        ax = plt.subplot(3, 6, i + 1)
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'{labels[i]} {i+1}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Part 2: Show the BoVW histograms
    colors = ['blue'] * 3 + ['red'] * 3
    for i in range(6):
        ax = plt.subplot(3, 6, i + 7)
        ax.bar(range(len(histograms[i])), histograms[i], 
               color=colors[i], alpha=0.7, edgecolor='black')
        ax.set_ylim([0, max(max(h) for h in histograms) * 1.1])
        ax.set_xlabel('Visual Word', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_title(f'Histogram {i+1}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Add comparison of similar images
    ax = plt.subplot(3, 2, 5)
    ax.bar(range(len(histograms[0])), histograms[0], 
           color='blue', alpha=0.6, label='Circle 1')
    ax.bar(range(len(histograms[1])), histograms[1], 
           color='cyan', alpha=0.6, label='Circle 2')
    ax.set_xlabel('Visual Word Index', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('SIMILAR Images (Both Circles)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = plt.subplot(3, 2, 6)
    ax.bar(range(len(histograms[0])), histograms[0], 
           color='blue', alpha=0.6, label='Circle 1')
    ax.bar(range(len(histograms[3])), histograms[3], 
           color='red', alpha=0.6, label='Square 1')
    ax.set_xlabel('Visual Word Index', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('DIFFERENT Images (Circle vs Square)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Bag of Visual Words - Complete Pipeline Demonstration', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('bovw_demonstration.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'bovw_demonstration.png'")
    print()
    
    plt.show()
    
    # Print histogram comparison
    print("HISTOGRAM COMPARISON:")
    print("-" * 70)
    
    # Compare Circle 1 vs Circle 2 (should be similar)
    circle1 = histograms[0]
    circle2 = histograms[1]
    similarity_circles = np.dot(circle1, circle2) / (np.linalg.norm(circle1) * np.linalg.norm(circle2))
    print(f"Similarity (Circle 1 vs Circle 2): {similarity_circles:.4f}")
    
    # Compare Circle 1 vs Square 1 (should be different)
    square1 = histograms[3]
    similarity_diff = np.dot(circle1, square1) / (np.linalg.norm(circle1) * np.linalg.norm(square1))
    print(f"Similarity (Circle 1 vs Square 1): {similarity_diff:.4f}")
    
    print()
    print("OBSERVATION:")
    print(f"  • Similar images (circles) have similarity: {similarity_circles:.4f}")
    print(f"  • Different images (circle vs square) have similarity: {similarity_diff:.4f}")
    print(f"  • Higher value = more similar")
    print()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_bovw()
    
    print("=" * 70)
    print("NEXT STEPS FOR STUDENTS:")
    print("=" * 70)
    print()
    print("Now you will complete the lab tasks:")
    print()
    print("TASK 1: Build Vocabulary")
    print("  • Use 20+ real images")
    print("  • Extract SIFT features")
    print("  • Create vocabulary with K=50 or K=100")
    print("  • Save as vocabulary.pkl")
    print()
    print("TASK 2: Image Representation")
    print("  • Load your vocabulary")
    print("  • Convert test images to histograms")
    print("  • Compare histograms between categories")
    print()
    print("TASK 3: Classification")
    print("  • Convert all images to histograms")
    print("  • Train KNN or SVM classifier")
    print("  • Evaluate with accuracy and confusion matrix")
    print()
    print("Good luck with your lab!")
    print("=" * 70)
