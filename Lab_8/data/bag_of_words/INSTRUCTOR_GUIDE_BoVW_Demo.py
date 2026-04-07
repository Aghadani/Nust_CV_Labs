"""
=============================================================================
INSTRUCTOR'S GUIDE: 10-MINUTE BAG OF VISUAL WORDS DEMONSTRATION
=============================================================================

Course: CS-474 Computer Vision
Lab: Feature Extraction using Bag of Visual Words
Duration: 10 minutes
File: bovw_10min_demo.py

=============================================================================
PURPOSE
=============================================================================

This demonstration code is designed to:
1. Show students the COMPLETE BoVW pipeline in one simple example
2. Build intuition before students tackle their lab tasks
3. Clarify the relationship between features, vocabulary, and histograms
4. Demonstrate that similar images produce similar histograms

=============================================================================
TEACHING SCRIPT (10 MINUTES)
=============================================================================

MINUTE 0-1: INTRODUCTION
------------------------
"Today we're learning Bag of Visual Words. I'm going to show you the entire
pipeline in 10 minutes using simple examples. Then you'll implement it 
yourself with real images."

"Think of it like this: In NLP, we represent documents by word frequencies.
In computer vision, we represent images by 'visual word' frequencies."


MINUTE 1-2: SHOW THE CODE STRUCTURE
------------------------------------
[Open bovw_10min_demo.py and scroll through]

"Notice the code has 5 clear steps:
1. Create sample images (in your lab, you'll use real images)
2. Extract SIFT features from ALL images
3. Create vocabulary using K-Means
4. Convert each image to a histogram
5. Visualize and compare"

"Let's run it and see what happens..."


MINUTE 2-3: RUN STEP 1 & 2
---------------------------
[Run the code, pause after Step 2]

"STEP 1: I created 6 simple images - 3 with circles, 3 with squares.
These are like your 'cars' and 'flowers' categories in the lab.

STEP 2: SIFT found keypoints in each image. See these numbers?
  Image 1: 24 keypoints, (24, 128) descriptors
  
What does (24, 128) mean?
  • 24 = number of interesting points SIFT found
  • 128 = dimensions of each descriptor (SIFT always uses 128)

Total descriptors: Let's say 150 from all 6 images.
These 150 descriptors are like puzzle pieces from all images mixed together."


MINUTE 3-5: EXPLAIN STEP 3 (K-MEANS)
-------------------------------------
[Point to the K-Means section]

"STEP 3: This is the magic step - creating the vocabulary.

We take all 150 descriptors and cluster them into K=10 groups using K-Means.
Why 10? I chose a small number so it's easy to visualize.
In your lab, you'll use K=50 or K=100.

The 10 cluster centers become our 'visual words'.
Think of them as common patterns that appear in images:
  • Visual word 1 might represent 'horizontal edges'
  • Visual word 2 might represent 'circular shapes'
  • Visual word 3 might represent 'corners'

Vocabulary shape: (10, 128)
  • 10 visual words
  • Each described by 128 numbers

This vocabulary.pkl file is what you'll save in Task 1!"


MINUTE 5-7: EXPLAIN STEP 4 (HISTOGRAMS)
----------------------------------------
[Point to histogram creation]

"STEP 4: Now we convert each image to a histogram.

For Image 1 (a circle image):
1. Extract its SIFT features - say 24 features
2. For each feature, find the NEAREST visual word in vocabulary
   Feature 1 → closest to Visual Word 3
   Feature 2 → closest to Visual Word 7
   Feature 3 → closest to Visual Word 3
   ...and so on
3. Count occurrences: How many mapped to Word 1? Word 2? etc.
4. Normalize so histogram sums to 1.0

Result: A 10-number histogram representing the image!
  [0.15, 0.05, 0.20, 0.10, 0.0, 0.05, 0.25, 0.10, 0.05, 0.05]

Every image, regardless of size, becomes a 10-number vector!"


MINUTE 7-9: SHOW THE VISUALIZATION
-----------------------------------
[The visualization should now be displayed]

"Look at this visualization:

TOP ROW: The 6 original images
  • First 3 are circles (blue)
  • Last 3 are squares (red)

MIDDLE ROW: The BoVW histograms for each image
  • Notice images 1, 2, 3 (all circles) have similar histogram shapes
  • Images 4, 5, 6 (all squares) have different histogram shapes

BOTTOM LEFT: Comparing two circle images
  • The histograms overlap a lot
  • Similarity score: ~0.85 (high)

BOTTOM RIGHT: Comparing circle vs square
  • The histograms look very different
  • Similarity score: ~0.35 (low)

THIS IS WHY BOVW WORKS!
Similar images → similar histograms → classifier can learn to distinguish!"


MINUTE 9-10: CONNECT TO LAB TASKS
----------------------------------
"Now let's connect this to your lab tasks:

TASK 1 - Build Vocabulary:
  • You'll use 20+ REAL images (cars, flowers)
  • Extract SIFT from all of them
  • K-Means with K=50 or K=100 (more visual words = better)
  • Save vocabulary.pkl

TASK 2 - Image Representation:
  • Load the vocabulary
  • Convert test images to histograms
  • Compare histograms from different categories

TASK 3 - Classification:
  • Convert ALL your images to histograms
  • Train KNN or SVM on these histograms
  • Test and evaluate accuracy

Questions before we start the lab?"


=============================================================================
KEY POINTS TO EMPHASIZE
=============================================================================

1. SIFT extracts local features (descriptors) from images
   ↓
2. K-Means clusters ALL features into K "visual words"
   ↓
3. Each image is represented by a HISTOGRAM of visual word frequencies
   ↓
4. Similar images have similar histograms
   ↓
5. We can train classifiers on these histograms!


=============================================================================
COMMON STUDENT QUESTIONS & ANSWERS
=============================================================================

Q: "Why do we need K-Means? Can't we just use SIFT features directly?"
A: "SIFT gives different numbers of features for different images. We need
    a fixed-length representation for classification. K-Means gives us that."

Q: "What's a good value for K?"
A: "Start with 50-100. Too small (K=5) loses detail. Too large (K=1000) can
    overfit. Experiment and see what works best!"

Q: "Why 128 dimensions for SIFT?"
A: "That's how SIFT works - it describes each keypoint using 128 numbers
    based on gradient orientations. It's a fixed standard."

Q: "Do all images need the same number of SIFT features?"
A: "No! That's the beauty. One image might have 100 features, another 200.
    But both become the SAME size histogram (K bins)."

Q: "What if an image has no SIFT features?"
A: "We return a zero histogram. The image had no interesting patterns.
    Use better/clearer images."


=============================================================================
TROUBLESHOOTING
=============================================================================

If visualization doesn't show:
  • Make sure matplotlib is installed: pip install matplotlib
  • Check if running in environment that supports GUI
  • Try adding plt.show() manually

If SIFT not available:
  • Need opencv-contrib-python: pip install opencv-contrib-python
  • Standard opencv-python doesn't include SIFT

If K-Means is slow:
  • Normal for large datasets
  • In demo, K=10 with 150 descriptors is instant
  • In real lab, K=100 with 5000+ descriptors takes 1-2 minutes


=============================================================================
AFTER THE DEMONSTRATION
=============================================================================

1. Ask students: "What are the 4 main steps of BoVW?"
   Expected: Feature extraction, Vocabulary building, Vector quantization, 
             Histogram creation

2. Ask: "Why do similar images have similar histograms?"
   Expected: Because they contain similar local patterns/features

3. Distribute the lab tasks and let students begin

4. Remind them they can refer to this demo code if they get stuck


=============================================================================
FILES GENERATED BY DEMO
=============================================================================

1. demo_vocabulary.pkl - The visual vocabulary (can show students this file)
2. bovw_demonstration.png - The visualization (great for lab reports!)


=============================================================================
TIMING BREAKDOWN
=============================================================================

0:00 - 1:00   Introduction and code overview
1:00 - 3:00   Run and explain Steps 1-2 (images and features)
3:00 - 5:00   Explain Step 3 (K-Means vocabulary)
5:00 - 7:00   Explain Step 4 (histogram creation)
7:00 - 9:00   Show visualization and results
9:00 - 10:00  Connect to lab tasks and Q&A

Total: 10 minutes


=============================================================================
GOOD LUCK TEACHING!
=============================================================================
"""

if __name__ == "__main__":
    print(__doc__)
