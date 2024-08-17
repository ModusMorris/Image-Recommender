# Image Recommender System
## Overview
This project is an advanced image recommendation system that leverages various techniques from computer vision and machine learning. The goal is to find visually similar images based on their content and color profiles. The system uses multiple approaches to analyze and compare images, including embedding-based methods, color histogram analysis, and dimensionality reduction techniques.

## Key Features
### 1. Embedding-Based Image Recommendation
In the **'embedding_recommender.py'** file, a pre-trained **ResNet50 model** is used to generate **image embeddings**. An embedding is a condensed representation of an image in a high-dimensional space where similar images are located close to each other.

**Image Preprocessing:** Images are resized and normalized to a fixed size before being fed into the ResNet50 model.
**Feature Extraction:** The ResNet50 model, originally trained for image classification, is modified by removing the final classification layer. This allows the model to function as a feature extractor, converting an image into a feature vector.
**Similarity Calculation:** The similarity between images is computed using various metrics:
**Cosine Similarity:** Measures the angle between vectors, indicating how similar the direction of the vectors is.
**Euclidean Distance:** Computes the straight-line distance between two points (embeddings) in the feature space.
**Manhattan Distance (Cityblock Distance):** Measures the distance along the axes of the space, which can be more robust to outliers in individual dimensions.
### 2. Color Analysis and Profiling
The **'color_profiling.py'** file focuses on analyzing the **color profiles** of images to find visually similar images based on their color compositions.

**RGB Histogram Creation:** For each image, an RGB histogram is created, representing the frequency of different color values (Red, Green, Blue) in the image. These histograms are normalized to ensure comparability across images of different sizes.
**Chi-Square Distance:** This method is used to measure the similarity between two color profiles. It calculates the squared difference between the histograms, which provides a measure of the discrepancy between the color distributions of two images.
**Parallel Processing:** To increase efficiency, histogram computation for large image datasets is parallelized by utilizing multiple CPU cores.
### 3. Dimensionality Reduction and Similarity Calculation
The **'dimension_reduction_similarity.py'** file implements a technique for dimensionality reduction of image data using **Principal Component Analysis (PCA)** and then uses these reduced representations to calculate image similarities.

**PCA (Principal Component Analysis):** PCA is a statistical method used to reduce the dimensionality of data while retaining as much variance (information) as possible. In this context, PCA is used to reduce the size of the color histograms, which improves the efficiency of similarity calculations.
**Cosine Similarity:** After dimensionality reduction, cosine similarity is used to determine the closeness of the reduced vectors (histograms). This method is particularly suitable for measuring similarity in high-dimensional spaces.
### 4. Visualization of Results
Both modules (**'color_profiling.py'** and **'dimension_reduction_similarity.py'**) include functions to visualize the results. Found similar images can be displayed in a grid format, allowing for an intuitive visual assessment of similarity.

## Directory Structure
- **embedding_recommender.py:** Implements the image recommendation system based on embeddings generated using a pre-trained ResNet50 model.
- **color_profiling.py:** Analyzes color histograms of images and finds similar images based on their color profiles.
- **dimension_reduction_similarity.py:** Performs dimensionality reduction on color histograms and computes similarities based on the reduced representations.

## Installation
1. Clone the repository:
`git clone https://github.com/YourUsername/YourRepository.git`
2. Install the required Python packages:
`pip install -r requirements.txt`


## Usage
### **1. Embedding-Based Recommendations:**
Run **'embedding_recommender.py'** to find similar images based on embeddings. You can input a new image, compute its embedding, and find the most similar images in your dataset.

### **2. Color Analysis:**
Use **'color_profiling.py'** to analyze the color histograms of images. This method is particularly useful when searching for images with similar color compositions.

### **3. Dimensionality Reduction and Similarity Calculation:**
Utilize **'dimension_reduction_similarity.py'** to reduce the dimensionality of color histograms and find similar images based on the cosine similarity of the reduced data.

## Dependencies
- Python 3.x
- torch
- torchvision
- scikit-learn
- PIL
- matplotlib
- numba
- tqdm
