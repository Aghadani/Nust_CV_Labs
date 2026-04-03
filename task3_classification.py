"""
Lab 08 - Task 3: Image Classification using BoVW
=================================================

This script demonstrates how to use BoVW representations for image classification.

Learning Objectives:
- Convert multiple images to BoVW histograms
- Split data into training and testing sets
- Train classifiers (KNN and SVM) on BoVW features
- Evaluate model performance with metrics
- Visualize results with confusion matrix

Author: Computer Vision Lab
Course: CS-474
"""

import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from scipy.spatial.distance import cdist

def load_vocabulary(vocab_file='vocabulary.pkl'):
    """Load the vocabulary created in Task 1."""
    with open(vocab_file, 'rb') as f:
        vocabulary = pickle.load(f)
    return vocabulary

def extract_sift_from_image(image_path):
    """Extract SIFT descriptors from an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def create_bow_histogram(descriptors, vocabulary, normalize=True):
    """Create BoVW histogram from descriptors."""
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(len(vocabulary))
    
    # Vector quantization: find nearest visual word for each descriptor
    distances = cdist(descriptors, vocabulary, metric='euclidean')
    word_indices = np.argmin(distances, axis=1)
    
    # Create histogram
    histogram, _ = np.histogram(word_indices, 
                                bins=len(vocabulary), 
                                range=(0, len(vocabulary)))
    histogram = histogram.astype(float)
    
    # Normalize
    if normalize:
        total = np.sum(histogram)
        if total > 0:
            histogram = histogram / total
    
    return histogram

def load_dataset_from_folders(dataset_path, vocabulary):
    """
    Load images from organized folders and convert to BoVW histograms.
    
    Expected folder structure:
    dataset_path/
        class_0/
            image1.jpg
            image2.jpg
            ...
        class_1/
            image1.jpg
            image2.jpg
            ...
    
    Args:
        dataset_path: Path to dataset root folder
        vocabulary: Visual vocabulary
        
    Returns:
        X: Feature matrix (N x K) where N=images, K=vocabulary size
        y: Labels (N,)
        class_names: List of class names
    """
    print("Loading dataset from folders...")
    print(f"Dataset path: {dataset_path}")
    print()
    
    X = []  # Feature vectors
    y = []  # Labels
    class_names = []
    
    # Get list of class folders
    class_folders = sorted([f for f in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, f))])
    
    if len(class_folders) == 0:
        print("Error: No class folders found!")
        return None, None, None
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    print()
    
    # Process each class
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(dataset_path, class_name)
        class_names.append(class_name)
        
        # Get images in this class
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(class_path) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        print(f"Class {class_idx} ({class_name}): {len(image_files)} images")
        
        # Process each image
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            # Extract SIFT and create BoVW histogram
            descriptors = extract_sift_from_image(img_path)
            histogram = create_bow_histogram(descriptors, vocabulary)
            
            X.append(histogram)
            y.append(class_idx)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print()
    print(f"Dataset loaded successfully!")
    print(f"  Total images: {len(X)}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Number of classes: {len(class_names)}")
    print()
    
    return X, y, class_names

def load_dataset_manual(image_paths, labels, vocabulary):
    """
    Alternative method: manually specify image paths and labels.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels (0, 1, 2, ...)
        vocabulary: Visual vocabulary
        
    Returns:
        X: Feature matrix
        y: Labels
    """
    X = []
    y = []
    
    print(f"Loading {len(image_paths)} images...")
    
    for img_path, label in zip(image_paths, labels):
        descriptors = extract_sift_from_image(img_path)
        histogram = create_bow_histogram(descriptors, vocabulary)
        X.append(histogram)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset created: {X.shape[0]} images with {X.shape[1]} features each")
    return X, y

def train_and_evaluate_knn(X_train, X_test, y_train, y_test, k=3):
    """
    Train and evaluate K-Nearest Neighbors classifier.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        k: Number of neighbors
        
    Returns:
        model: Trained KNN model
        y_pred: Predictions on test set
        accuracy: Accuracy score
    """
    print("\n" + "=" * 60)
    print("Training K-Nearest Neighbors (KNN) Classifier")
    print("=" * 60)
    print(f"Parameters: k={k}")
    print()
    
    # Create and train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training completed!")
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    
    return knn, y_pred, accuracy

def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Linear SVM classifier.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        
    Returns:
        model: Trained SVM model
        y_pred: Predictions on test set
        accuracy: Accuracy score
    """
    print("\n" + "=" * 60)
    print("Training Linear Support Vector Machine (SVM)")
    print("=" * 60)
    print()
    
    # Create and train Linear SVM
    svm = LinearSVC(random_state=42, max_iter=1000)
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training completed!")
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    
    return svm, y_pred, accuracy

def plot_confusion_matrix(y_test, y_pred, class_names, title, save_name):
    """
    Plot confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        save_name: Filename to save plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot using seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='black')
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add accuracy text
    accuracy = accuracy_score(y_test, y_pred)
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy * 100:.2f}%',
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved as: {save_name}")
    plt.show()

def display_classification_report(y_test, y_pred, class_names):
    """
    Display detailed classification metrics.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    print("\n" + "=" * 60)
    print("Detailed Classification Report")
    print("=" * 60)
    print()
    
    report = classification_report(y_test, y_pred, 
                                   target_names=class_names,
                                   digits=4)
    print(report)

def compare_classifiers(results):
    """
    Compare performance of different classifiers.
    
    Args:
        results: Dictionary with classifier names and their accuracies
    """
    print("\n" + "=" * 60)
    print("Classifier Comparison")
    print("=" * 60)
    print()
    
    # Display table
    print(f"{'Classifier':<20} {'Accuracy':<10}")
    print("-" * 30)
    for name, acc in results.items():
        print(f"{name:<20} {acc * 100:>6.2f}%")
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    classifiers = list(results.keys())
    accuracies = [results[c] * 100 for c in classifiers]
    
    bars = plt.bar(classifiers, accuracies, color=['steelblue', 'coral'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Classifier Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('classifier_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as: classifier_comparison.png")
    plt.show()

def visualize_feature_space(X, y, class_names):
    """
    Visualize the feature space using PCA for dimensionality reduction.
    
    Args:
        X: Feature matrix
        y: Labels
        class_names: Class names
    """
    from sklearn.decomposition import PCA
    
    print("\nVisualizing feature space with PCA...")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for class_idx, class_name in enumerate(class_names):
        mask = y == class_idx
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[class_idx], label=class_name,
                   alpha=0.7, s=100, edgecolors='black', linewidth=1)
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
              fontsize=12)
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
              fontsize=12)
    plt.title('BoVW Feature Space Visualization (PCA)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_space_pca.png', dpi=150, bbox_inches='tight')
    print("Feature space visualization saved as: feature_space_pca.png")
    plt.show()

def main():
    """
    Main function to execute Task 3.
    """
    print("=" * 60)
    print("TASK 3: Image Classification using BoVW")
    print("=" * 60)
    print()
    
    # ========================================================================
    # CONFIGURATION - Students should modify these paths
    # ========================================================================
    
    # Path to vocabulary file
    VOCAB_FILE = 'vocabulary.pkl'
    
    # Path to dataset folder with organized subdirectories
    # For folder-based loading:
    DATASET_PATH = './dataset'
    
    # Alternative: Manual specification (comment out if using folder structure)
    # MANUAL_MODE = False
    # IMAGE_PATHS = ['./images/car1.jpg', './images/car2.jpg', ...]
    # LABELS = [0, 0, 1, 1, ...]  # 0=car, 1=flower, etc.
    
    # Train-test split ratio
    TEST_SIZE = 0.2  # 20% for testing, 80% for training
    
    # Classifier parameters
    KNN_K = 3  # Number of neighbors for KNN
    
    # ========================================================================
    
    # Load vocabulary
    print("Loading vocabulary...")
    vocabulary = load_vocabulary(VOCAB_FILE)
    print(f"Vocabulary shape: {vocabulary.shape}")
    print()
    
    # Load dataset
    X, y, class_names = load_dataset_from_folders(DATASET_PATH, vocabulary)
    
    if X is None:
        print("\nError loading dataset. Please check your folder structure.")
        print("\nExpected structure:")
        print("dataset/")
        print("  cars/")
        print("    car_001.jpg")
        print("    car_002.jpg")
        print("  flowers/")
        print("    flower_001.jpg")
        print("    flower_002.jpg")
        return
    
    # Split into training and testing sets
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} images")
    print(f"  Test set: {len(X_test)} images")
    print()
    
    # Visualize feature space
    visualize_feature_space(X, y, class_names)
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate KNN
    knn_model, knn_pred, knn_acc = train_and_evaluate_knn(
        X_train, X_test, y_train, y_test, k=KNN_K
    )
    results['KNN'] = knn_acc
    
    # Display classification report
    display_classification_report(y_test, knn_pred, class_names)
    
    # Plot confusion matrix for KNN
    plot_confusion_matrix(y_test, knn_pred, class_names,
                         f'Confusion Matrix - KNN (k={KNN_K})',
                         'confusion_matrix_knn.png')
    
    # Train and evaluate SVM
    svm_model, svm_pred, svm_acc = train_and_evaluate_svm(
        X_train, X_test, y_train, y_test
    )
    results['Linear SVM'] = svm_acc
    
    # Display classification report
    display_classification_report(y_test, svm_pred, class_names)
    
    # Plot confusion matrix for SVM
    plot_confusion_matrix(y_test, svm_pred, class_names,
                         'Confusion Matrix - Linear SVM',
                         'confusion_matrix_svm.png')
    
    # Compare classifiers
    compare_classifiers(results)
    
    print("\n" + "=" * 60)
    print("TASK 3 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated outputs:")
    print("1. confusion_matrix_knn.png - KNN confusion matrix")
    print("2. confusion_matrix_svm.png - SVM confusion matrix")
    print("3. classifier_comparison.png - Performance comparison")
    print("4. feature_space_pca.png - 2D visualization of features")
    print()
    print("Key Observations:")
    print(f"  - Best classifier: {max(results, key=results.get)} "
          f"with {max(results.values()) * 100:.2f}% accuracy")
    print(f"  - Dataset size: {len(X)} images, {len(class_names)} classes")
    print(f"  - Feature dimension: {X.shape[1]} (vocabulary size)")
    print()
    print("Analysis Questions:")
    print("1. How does the vocabulary size (K) affect classification accuracy?")
    print("2. Which classifier works better for your dataset and why?")
    print("3. What does the confusion matrix tell you about misclassifications?")
    print("4. How does BoVW compare to using raw pixels as features?")
    print()

if __name__ == "__main__":
    main()
