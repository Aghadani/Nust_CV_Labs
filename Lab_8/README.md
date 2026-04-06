# Lab 08: Bag of Visual Words (BoVW)
## Feature Extraction for Image Classification

**Course:** CS-474 Computer Vision  
**Semester:** 6th  
**Lab Topic:** Feature Extraction using Bag of Visual Words

---

## 📁 Files Included

### Lab Manual
- `Lab_08_Bag_of_Visual_Words.docx` - Complete lab manual with instructions

### Python Scripts
- `task1_build_vocabulary.py` - Build visual vocabulary from training images
- `task2_image_representation.py` - Convert images to BoVW histograms
- `task3_classification.py` - Classify images using BoVW features

### Utilities
- `setup_verification.py` - Verify your setup before starting
- `STUDENT_GUIDE.txt` - Comprehensive student guide

---

## 🎯 Learning Objectives

After completing this lab, you will:

1. Understand the Bag of Visual Words concept and pipeline
2. Extract SIFT features from images
3. Create a visual vocabulary using K-Means clustering
4. Represent images as histograms of visual words
5. Build and evaluate image classifiers
6. Compare different classification algorithms (KNN vs SVM)

---

## 📋 Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of arrays and matrices (NumPy)
- Familiarity with machine learning concepts
- Basic understanding of image processing

### Required Software
- Python 3.7 or higher
- pip (Python package manager)

### Required Python Packages
```bash
pip install opencv-python opencv-contrib-python numpy scikit-learn matplotlib seaborn scipy
```

---

## 🚀 Quick Start Guide

### Step 1: Verify Your Setup
Run the verification script to check if everything is installed correctly:

```bash
python setup_verification.py
```

This will:
- Check all required packages
- Create necessary folders
- Test SIFT and K-Means functionality

### Step 2: Prepare Your Dataset

Organize your images in the following structure:

```
training_images/          # For Task 1
    car_001.jpg
    car_002.jpg
    ...
    flower_001.jpg
    flower_002.jpg
    ...

test_images/              # For Task 2
    car_test.jpg
    flower_test.jpg

dataset/                  # For Task 3
    cars/
        car_001.jpg
        car_002.jpg
        ...
    flowers/
        flower_001.jpg
        flower_002.jpg
        ...
```

**Minimum Requirements:**
- 20 images for training (10 per category)
- 2 images for testing
- 20+ images in organized folders for classification

### Step 3: Run the Tasks

#### Task 1: Build Visual Vocabulary
```bash
python task1_build_vocabulary.py
```

**What it does:**
- Extracts SIFT features from all training images
- Applies K-Means clustering
- Creates visual vocabulary (codebook)
- Saves `vocabulary.pkl`

**Output:**
- `vocabulary.pkl` - The visual vocabulary
- `vocabulary_statistics.png` - Visualization of vocabulary

**Time:** ~5-10 minutes

---

#### Task 2: Image Representation
```bash
python task2_image_representation.py
```

**What it does:**
- Loads the vocabulary from Task 1
- Converts images to BoVW histograms
- Visualizes and compares histograms

**Output:**
- `histogram_car.png` - Histogram for car image
- `histogram_flower.png` - Histogram for flower image
- `histogram_comparison.png` - Side-by-side comparison

**Time:** ~3-5 minutes

---

#### Task 3: Image Classification
```bash
python task3_classification.py
```

**What it does:**
- Converts dataset to BoVW representations
- Splits data into train/test sets
- Trains KNN and SVM classifiers
- Evaluates performance

**Output:**
- `confusion_matrix_knn.png` - KNN results
- `confusion_matrix_svm.png` - SVM results
- `classifier_comparison.png` - Performance comparison
- `feature_space_pca.png` - 2D visualization

**Time:** ~10-15 minutes

---

## 🔧 Configuration

Each script has a CONFIGURATION section that you need to modify:

### Example from task1_build_vocabulary.py:
```python
# ========================================================================
# CONFIGURATION - Students should modify these paths
# ========================================================================

IMAGE_FOLDER = './training_images'  # Change to your folder path
NUM_CLUSTERS = 50                   # Try 50, 100, 200
OUTPUT_FILE = 'vocabulary.pkl'      # Output filename
```

**Important:** Always use the correct paths for your system!

---

## 📊 Understanding the Pipeline

### The BoVW Pipeline

```
Step 1: Feature Extraction
├── Read image
├── Convert to grayscale
├── Detect SIFT keypoints
└── Extract 128-dim descriptors
           ↓
Step 2: Vocabulary Building
├── Collect all descriptors
├── Apply K-Means clustering
└── Save cluster centers (visual words)
           ↓
Step 3: Vector Quantization
├── Extract descriptors from new image
├── Find nearest visual word for each
└── Create histogram of word occurrences
           ↓
Step 4: Classification
├── Convert all images to histograms
├── Train classifier (KNN or SVM)
└── Evaluate on test set
```

---

## 📈 Expected Results

### Typical Performance Metrics

With a well-prepared dataset:
- **Accuracy:** 70-95% (depends on dataset quality)
- **Vocabulary Size (K):** 50-200 for small datasets
- **Features per Image:** 100-500 SIFT keypoints
- **Processing Time:** Few minutes for small datasets

### Factors Affecting Performance

**Positive Impact:**
- More training images
- Diverse image variations
- Appropriate vocabulary size (K)
- Clear, high-quality images

**Negative Impact:**
- Too few training images
- Very similar categories
- Blurry or low-resolution images
- Incorrect K value (too small or too large)

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "No module named 'cv2'"
```bash
pip install opencv-python opencv-contrib-python
```

#### 2. "SIFT is not available"
```bash
pip install opencv-contrib-python --upgrade
```

#### 3. "No images found in folder"
- Check folder path (use absolute path if needed)
- Verify image extensions (.jpg, .png, .bmp)
- Check file permissions

#### 4. "No SIFT features extracted"
- Use larger images (minimum 300x300)
- Ensure images have textures/edges
- Check image quality (not too blurry)

#### 5. "Low classification accuracy"
- Add more training images (20+ per class)
- Increase vocabulary size (try K=100 or K=200)
- Use more diverse training data
- Check that categories are visually distinct

#### 6. "KMeans not converging"
- Reduce K (number of clusters)
- Increase max_iter in KMeans
- Verify sufficient descriptors collected

---

## 📝 Lab Report Guidelines

Your lab report should include:

### 1. Pre-lab Assignment (2 marks)
- Fill in the blanks answers
- Conceptual question responses

### 2. Code Implementation (4 marks)
- All three task scripts
- Proper comments and documentation
- Correct configuration

### 3. Results and Analysis (2 marks)
- All generated plots and visualizations
- Accuracy scores and confusion matrices
- Comparison of classifiers
- Analysis of results:
  - Which classifier performed better and why?
  - How does K affect performance?
  - What patterns appear in misclassifications?

### 4. Viva Questions (2 marks)
Be prepared to answer:
- Explain the BoVW pipeline
- What is vector quantization?
- Why normalize histograms?
- How does SIFT work?
- What is the role of K in K-Means?
- Advantages and limitations of BoVW

---

## 🎓 Learning Resources

### Recommended Reading

1. **SIFT Paper:**
   - Lowe, D. G. (1999). "Object recognition from local scale-invariant features"

2. **BoVW Paper:**
   - Csurka, G., et al. (2004). "Visual categorization with bags of keypoints"

3. **Online Tutorials:**
   - OpenCV SIFT Tutorial: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
   - Scikit-learn Documentation: https://scikit-learn.org/

### Video Tutorials
- Search for "Bag of Visual Words tutorial" on YouTube
- Look for "SIFT feature extraction explained"

---

## 🔬 Extensions and Experiments

After completing the basic lab, try these challenges:

### Easy
1. **Different Categories:**
   - Add a third category (e.g., bicycles)
   - See how accuracy changes with 3+ classes

2. **Vocabulary Size Experiment:**
   - Test K = 25, 50, 100, 200, 500
   - Plot accuracy vs K

### Medium
3. **Different Features:**
   - Replace SIFT with ORB or AKAZE
   - Compare performance and speed

4. **Parameter Tuning:**
   - Experiment with different KNN k values
   - Try different SVM kernels (RBF, polynomial)

### Advanced
5. **Spatial Pyramid Matching:**
   - Divide image into grid
   - Create histogram for each cell
   - Concatenate for richer representation

6. **TF-IDF Weighting:**
   - Apply term frequency - inverse document frequency
   - Give more weight to discriminative visual words

---

## 💡 Tips for Success

### Before the Lab
✅ Read the student guide thoroughly  
✅ Install all required packages  
✅ Prepare your dataset (20+ images per category)  
✅ Run setup_verification.py  

### During the Lab
✅ Read code comments carefully  
✅ Start with small K (50) to test quickly  
✅ Save all outputs for your report  
✅ Print intermediate results to debug  
✅ Ask questions if stuck  

### For the Report
✅ Include all visualizations  
✅ Explain your results, don't just show numbers  
✅ Compare different approaches  
✅ Discuss what worked and what didn't  
✅ Suggest improvements  

---

## 📞 Support

If you encounter issues:

1. **Check the Student Guide** (`STUDENT_GUIDE.txt`)
2. **Review error messages** carefully
3. **Run verification script** to check setup
4. **Consult with lab instructor**
5. **Discuss with classmates** (but write your own code!)

---

## ⚖️ Academic Integrity

- Understand the concepts, don't just copy code
- You may discuss ideas with classmates
- Write your own code and report
- Cite any external resources used
- Ask instructor if unsure about collaboration rules

---

## 📄 License

This lab material is for educational purposes only.  
CS-474 Computer Vision Course  
6th Semester

---

## 🎉 Conclusion

Bag of Visual Words is a fundamental technique in computer vision. By completing this lab, you've learned how to:

- Extract meaningful features from images
- Build compact image representations
- Apply machine learning for classification

These concepts form the foundation for understanding modern computer vision techniques, including deep learning approaches.

**Good luck with your lab!** 🚀

---

*Last Updated: April 2026*
