"""
Lab 08 - Task 1: Building the Visual Vocabulary
================================================

This script demonstrates how to build a visual vocabulary using SIFT features
and K-Means clustering for the Bag of Visual Words approach.

Learning Objectives:
- Extract SIFT features from multiple images
- Collect all descriptors into a single dataset
- Apply K-Means clustering to create visual words
- Save the vocabulary for later use

Author: Computer Vision Lab
Course: CS-474
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_sift_features(image_path):
    """
    Extract SIFT features from a single image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        descriptors: SIFT descriptors (Nx128 array) or None if no features found
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to grayscale (SIFT works on grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    # keypoints: locations of interesting points in the image
    # descriptors: 128-dimensional feature vectors describing each keypoint
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    print(f"  Found {len(keypoints)} keypoints in {os.path.basename(image_path)}")
    
    return descriptors

def build_vocabulary(image_folder, num_clusters=50):
    """
    Build a visual vocabulary from a collection of images.
    
    Pipeline:
    1. Extract SIFT descriptors from all images
    2. Collect all descriptors into one large array
    3. Apply K-Means clustering
    4. Return cluster centers as the vocabulary
    
    Args:
        image_folder: Path to folder containing training images
        num_clusters: Number of visual words (K in K-Means)
        
    Returns:
        vocabulary: Array of cluster centers (K x 128)
        all_descriptors: Combined descriptors from all images
    """
    print("=" * 60)
    print("TASK 1: Building Visual Vocabulary")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Image folder: {image_folder}")
    print(f"  Number of clusters (K): {num_clusters}")
    print()
    
    # List to store all descriptors from all images
    all_descriptors = []
    
    # Get list of image files (common image formats)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    if len(image_files) == 0:
        print(f"Error: No images found in {image_folder}")
        return None, None
    
    print(f"Found {len(image_files)} images in the folder")
    print("\nExtracting SIFT features from images...")
    print("-" * 60)
    
    # Extract SIFT descriptors from each image
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, filename)
        print(f"[{i}/{len(image_files)}] Processing: {filename}")
        
        descriptors = extract_sift_features(image_path)
        
        if descriptors is not None and len(descriptors) > 0:
            # Add this image's descriptors to our collection
            all_descriptors.append(descriptors)
    
    # Check if we collected any descriptors
    if len(all_descriptors) == 0:
        print("\nError: No SIFT features were extracted from any image!")
        return None, None
    
    # Combine all descriptors into a single array
    # vstack = vertical stack (stack arrays vertically)
    all_descriptors = np.vstack(all_descriptors)
    
    print("\n" + "=" * 60)
    print("Feature Extraction Summary:")
    print("=" * 60)
    print(f"Total descriptors collected: {all_descriptors.shape[0]}")
    print(f"Descriptor dimension: {all_descriptors.shape[1]} (should be 128 for SIFT)")
    print()
    
    # Apply K-Means clustering
    print("Applying K-Means clustering to create visual vocabulary...")
    print(f"This may take a few minutes for large datasets...")
    print()
    
    # K-Means parameters:
    # n_clusters: number of visual words to create
    # n_init: number of times K-Means will run with different initializations
    # max_iter: maximum iterations for convergence
    # random_state: for reproducibility
    kmeans = KMeans(n_clusters=num_clusters, 
                    n_init=10, 
                    max_iter=300, 
                    random_state=42,
                    verbose=1)
    
    # Fit K-Means to find cluster centers
    kmeans.fit(all_descriptors)
    
    # The cluster centers become our visual vocabulary
    vocabulary = kmeans.cluster_centers_
    
    print("\n" + "=" * 60)
    print("Vocabulary Creation Complete!")
    print("=" * 60)
    print(f"Vocabulary shape: {vocabulary.shape}")
    print(f"  Number of visual words: {vocabulary.shape[0]}")
    print(f"  Dimension of each word: {vocabulary.shape[1]}")
    print()
    
    return vocabulary, all_descriptors

def save_vocabulary(vocabulary, filename='vocabulary.pkl'):
    """
    Save the vocabulary to a pickle file for later use.
    
    Args:
        vocabulary: The visual vocabulary (cluster centers)
        filename: Name of the file to save
    """
    with open(filename, 'wb') as f:
        pickle.dump(vocabulary, f)
    print(f"Vocabulary saved to: {filename}")

def visualize_vocabulary_stats(vocabulary, all_descriptors, num_clusters):
    """
    Create visualizations to understand the vocabulary.
    
    Args:
        vocabulary: The cluster centers
        all_descriptors: All SIFT descriptors used
        num_clusters: Number of clusters
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Visual Vocabulary Statistics', fontsize=16, fontweight='bold')
    
    # 1. Distribution of first 5 dimensions of vocabulary
    ax1 = axes[0, 0]
    for i in range(min(5, vocabulary.shape[1])):
        ax1.hist(vocabulary[:, i], bins=30, alpha=0.5, label=f'Dim {i}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of First 5 Dimensions of Visual Words')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Heatmap of first 10 visual words (first 32 dimensions)
    ax2 = axes[0, 1]
    subset = vocabulary[:min(10, num_clusters), :32]
    im = ax2.imshow(subset, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Descriptor Dimensions (first 32)')
    ax2.set_ylabel('Visual Word Index')
    ax2.set_title('Heatmap of First 10 Visual Words')
    plt.colorbar(im, ax=ax2)
    
    # 3. Variance across visual words
    ax3 = axes[1, 0]
    variance_per_word = np.var(vocabulary, axis=1)
    ax3.bar(range(len(variance_per_word)), variance_per_word)
    ax3.set_xlabel('Visual Word Index')
    ax3.set_ylabel('Variance')
    ax3.set_title('Variance of Each Visual Word')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of descriptor norms
    ax4 = axes[1, 1]
    descriptor_norms = np.linalg.norm(all_descriptors, axis=1)
    vocab_norms = np.linalg.norm(vocabulary, axis=1)
    ax4.hist(descriptor_norms, bins=50, alpha=0.6, label='All Descriptors', color='blue')
    ax4.hist(vocab_norms, bins=30, alpha=0.6, label='Visual Words', color='red')
    ax4.set_xlabel('L2 Norm')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Descriptor Norms')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vocabulary_statistics.png', dpi=150, bbox_inches='tight')
    print("Vocabulary statistics plot saved as: vocabulary_statistics.png")
    plt.show()

def main():
    """
    Main function to execute Task 1.
    """
    # ========================================================================
    # CONFIGURATION - Students should modify these paths
    # ========================================================================
    
    # Path to folder containing your training images
    # For example: './images/training' or './dataset'
    IMAGE_FOLDER = './training_images'
    
    # Number of visual words to create
    # Start with 50 or 100, can experiment with larger values
    NUM_CLUSTERS = 50
    
    # Output filename for vocabulary
    OUTPUT_FILE = 'vocabulary.pkl'
    
    # ========================================================================
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' does not exist!")
        print("\nPlease create this folder and add at least 20 images.")
        print("Recommended structure:")
        print("  training_images/")
        print("    car_001.jpg")
        print("    car_002.jpg")
        print("    ...")
        print("    flower_001.jpg")
        print("    flower_002.jpg")
        print("    ...")
        return
    
    # Build the vocabulary
    vocabulary, all_descriptors = build_vocabulary(IMAGE_FOLDER, NUM_CLUSTERS)
    
    if vocabulary is None:
        print("\nFailed to build vocabulary. Please check your images.")
        return
    
    # Save the vocabulary
    save_vocabulary(vocabulary, OUTPUT_FILE)
    
    # Visualize vocabulary statistics
    print("\nGenerating vocabulary statistics...")
    visualize_vocabulary_stats(vocabulary, all_descriptors, NUM_CLUSTERS)
    
    print("\n" + "=" * 60)
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check that vocabulary.pkl was created")
    print("2. Review the vocabulary_statistics.png plot")
    print("3. Proceed to Task 2 for image representation")
    print()

if __name__ == "__main__":
    main()
