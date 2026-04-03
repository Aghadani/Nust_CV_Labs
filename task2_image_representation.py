"""
Lab 08 - Task 2: Image Representation using BoVW
=================================================

This script demonstrates how to represent images as histograms of visual words
using the vocabulary created in Task 1.

Learning Objectives:
- Load the pre-trained vocabulary
- Extract SIFT features from test images
- Perform vector quantization (map descriptors to visual words)
- Create and normalize histograms
- Compare histograms from different image categories

Author: Computer Vision Lab
Course: CS-474
"""

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

def load_vocabulary(vocab_file='vocabulary.pkl'):
    """
    Load the vocabulary created in Task 1.
    
    Args:
        vocab_file: Path to the vocabulary pickle file
        
    Returns:
        vocabulary: Array of visual words (cluster centers)
    """
    try:
        with open(vocab_file, 'rb') as f:
            vocabulary = pickle.load(f)
        print(f"Vocabulary loaded successfully!")
        print(f"  Shape: {vocabulary.shape}")
        print(f"  Number of visual words: {vocabulary.shape[0]}")
        return vocabulary
    except FileNotFoundError:
        print(f"Error: Vocabulary file '{vocab_file}' not found!")
        print("Please run Task 1 first to create the vocabulary.")
        return None

def extract_sift_from_image(image_path):
    """
    Extract SIFT descriptors from a single image.
    
    Args:
        image_path: Path to the image
        
    Returns:
        descriptors: SIFT descriptors (Nx128) or None
    """
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return descriptors

def vector_quantization(descriptors, vocabulary):
    """
    Map each descriptor to its nearest visual word.
    This is the core of the Bag of Visual Words approach.
    
    Args:
        descriptors: SIFT descriptors from an image (Nx128)
        vocabulary: Visual words (Kx128)
        
    Returns:
        word_indices: Index of nearest visual word for each descriptor
    """
    # Compute Euclidean distance between each descriptor and each visual word
    # cdist computes pairwise distances between two sets of points
    # Result shape: (num_descriptors, num_visual_words)
    distances = cdist(descriptors, vocabulary, metric='euclidean')
    
    # For each descriptor, find the index of the closest visual word
    # argmin finds the index of minimum value along axis 1 (across visual words)
    word_indices = np.argmin(distances, axis=1)
    
    return word_indices

def create_bow_histogram(descriptors, vocabulary, normalize=True):
    """
    Create a Bag of Visual Words histogram for an image.
    
    Pipeline:
    1. Map each descriptor to nearest visual word (vector quantization)
    2. Count occurrences of each visual word
    3. Normalize the histogram
    
    Args:
        descriptors: SIFT descriptors from the image
        vocabulary: Visual vocabulary
        normalize: Whether to normalize the histogram
        
    Returns:
        histogram: BoVW representation of the image
    """
    if descriptors is None or len(descriptors) == 0:
        # Return zero histogram if no features found
        return np.zeros(len(vocabulary))
    
    # Perform vector quantization
    word_indices = vector_quantization(descriptors, vocabulary)
    
    # Create histogram by counting occurrences of each visual word
    # bins: edges of histogram bins (0, 1, 2, ..., K)
    # range: the range of values to consider
    histogram, _ = np.histogram(word_indices, 
                                bins=len(vocabulary), 
                                range=(0, len(vocabulary)))
    
    # Convert to float for normalization
    histogram = histogram.astype(float)
    
    # Normalize the histogram
    if normalize:
        # L1 normalization: divide by sum (makes histogram sum to 1)
        total = np.sum(histogram)
        if total > 0:
            histogram = histogram / total
    
    return histogram

def visualize_bow_histogram(histogram, title, save_name=None):
    """
    Visualize a BoVW histogram.
    
    Args:
        histogram: The BoVW histogram to visualize
        title: Title for the plot
        save_name: Optional filename to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Create bar plot
    plt.bar(range(len(histogram)), histogram, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Visual Word Index', fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    max_freq = np.max(histogram)
    max_idx = np.argmax(histogram)
    nonzero_count = np.count_nonzero(histogram)
    
    stats_text = f'Max frequency: {max_freq:.4f} (word {max_idx})\n'
    stats_text += f'Non-zero words: {nonzero_count}/{len(histogram)}'
    
    plt.text(0.98, 0.97, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"Histogram saved as: {save_name}")
    
    plt.show()

def compare_histograms(hist1, hist2, label1, label2):
    """
    Compare two BoVW histograms side by side.
    
    Args:
        hist1, hist2: Histograms to compare
        label1, label2: Labels for the histograms
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Comparing BoVW Histograms from Different Categories', 
                 fontsize=16, fontweight='bold')
    
    # Plot first histogram
    axes[0].bar(range(len(hist1)), hist1, color='coral', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Visual Word Index', fontsize=12)
    axes[0].set_ylabel('Normalized Frequency', fontsize=12)
    axes[0].set_title(label1, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot second histogram
    axes[1].bar(range(len(hist2)), hist2, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Visual Word Index', fontsize=12)
    axes[1].set_ylabel('Normalized Frequency', fontsize=12)
    axes[1].set_title(label2, fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('histogram_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved as: histogram_comparison.png")
    plt.show()
    
    # Calculate and display similarity metrics
    print("\n" + "=" * 60)
    print("Histogram Comparison Metrics:")
    print("=" * 60)
    
    # Euclidean distance (lower = more similar)
    euclidean_dist = np.linalg.norm(hist1 - hist2)
    print(f"Euclidean Distance: {euclidean_dist:.4f}")
    
    # Cosine similarity (higher = more similar, range [-1, 1])
    dot_product = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    print(f"Cosine Similarity: {cosine_sim:.4f}")
    
    # Chi-square distance
    chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10)) / 2
    print(f"Chi-Square Distance: {chi_square:.4f}")
    
    print("\nInterpretation:")
    if euclidean_dist < 0.3:
        print("  - Very similar images (same category)")
    elif euclidean_dist < 0.6:
        print("  - Moderately similar images")
    else:
        print("  - Very different images (different categories)")
    print()

def analyze_histogram(histogram, image_name):
    """
    Provide detailed analysis of a histogram.
    
    Args:
        histogram: BoVW histogram
        image_name: Name of the image
    """
    print("\n" + "=" * 60)
    print(f"Histogram Analysis: {image_name}")
    print("=" * 60)
    
    # Basic statistics
    print(f"Histogram dimensions: {len(histogram)}")
    print(f"Sum of frequencies: {np.sum(histogram):.4f} (should be 1.0 if normalized)")
    print(f"Number of non-zero visual words: {np.count_nonzero(histogram)}")
    print(f"Percentage of active visual words: {100 * np.count_nonzero(histogram) / len(histogram):.1f}%")
    
    # Top visual words
    top_k = 5
    top_indices = np.argsort(histogram)[::-1][:top_k]
    print(f"\nTop {top_k} most frequent visual words:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Word {idx}: frequency = {histogram[idx]:.4f}")
    
    # Distribution statistics
    print(f"\nDistribution statistics:")
    print(f"  Mean frequency: {np.mean(histogram):.4f}")
    print(f"  Std deviation: {np.std(histogram):.4f}")
    print(f"  Max frequency: {np.max(histogram):.4f}")
    print(f"  Min frequency: {np.min(histogram):.4f}")
    print()

def process_image(image_path, vocabulary):
    """
    Complete pipeline to convert an image to BoVW representation.
    
    Args:
        image_path: Path to the image
        vocabulary: Visual vocabulary
        
    Returns:
        histogram: BoVW histogram
        descriptors: SIFT descriptors (for debugging)
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")
    print("-" * 60)
    
    # Extract SIFT features
    descriptors = extract_sift_from_image(image_path)
    
    if descriptors is None:
        print("  No features extracted!")
        return None, None
    
    print(f"  Extracted {len(descriptors)} SIFT descriptors")
    
    # Create BoVW histogram
    histogram = create_bow_histogram(descriptors, vocabulary, normalize=True)
    print(f"  Created histogram with {len(histogram)} bins")
    
    return histogram, descriptors

def main():
    """
    Main function to execute Task 2.
    """
    print("=" * 60)
    print("TASK 2: Image Representation using BoVW")
    print("=" * 60)
    print()
    
    # ========================================================================
    # CONFIGURATION - Students should modify these paths
    # ========================================================================
    
    # Path to vocabulary file from Task 1
    VOCAB_FILE = 'vocabulary.pkl'
    
    # Paths to test images (one from each category)
    IMAGE_CAR = './test_images/car_test.jpg'
    IMAGE_FLOWER = './test_images/flower_test.jpg'
    
    # ========================================================================
    
    # Load vocabulary
    vocabulary = load_vocabulary(VOCAB_FILE)
    
    if vocabulary is None:
        return
    
    print()
    
    # Process first image (Car)
    hist_car, desc_car = process_image(IMAGE_CAR, vocabulary)
    
    if hist_car is not None:
        analyze_histogram(hist_car, os.path.basename(IMAGE_CAR))
        visualize_bow_histogram(hist_car, 
                               f'BoVW Histogram: {os.path.basename(IMAGE_CAR)}',
                               'histogram_car.png')
    
    # Process second image (Flower)
    hist_flower, desc_flower = process_image(IMAGE_FLOWER, vocabulary)
    
    if hist_flower is not None:
        analyze_histogram(hist_flower, os.path.basename(IMAGE_FLOWER))
        visualize_bow_histogram(hist_flower, 
                               f'BoVW Histogram: {os.path.basename(IMAGE_FLOWER)}',
                               'histogram_flower.png')
    
    # Compare the two histograms
    if hist_car is not None and hist_flower is not None:
        print("\n" + "=" * 60)
        print("Comparing Images from Different Categories")
        print("=" * 60)
        compare_histograms(hist_car, hist_flower,
                          f'Car: {os.path.basename(IMAGE_CAR)}',
                          f'Flower: {os.path.basename(IMAGE_FLOWER)}')
    
    print("\n" + "=" * 60)
    print("TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated outputs:")
    print("1. histogram_car.png - BoVW histogram for car image")
    print("2. histogram_flower.png - BoVW histogram for flower image")
    print("3. histogram_comparison.png - Side-by-side comparison")
    print("\nObservations to note:")
    print("- Different categories should have different histogram patterns")
    print("- Similar categories will have similar histogram shapes")
    print("- The histogram is a compact, fixed-length representation")
    print("\nProceed to Task 3 for classification!")
    print()

if __name__ == "__main__":
    main()
