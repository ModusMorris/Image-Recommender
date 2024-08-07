import os
import pickle
import sqlite3
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def extract_histogram(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        histogram = np.array(img.histogram()).reshape((3, 256)).astype(float)
        histogram /= histogram.sum()
        return histogram.flatten()

def load_histograms(pickle_file):
    with open(pickle_file, "rb") as f:
        histograms = pickle.load(f)
    return histograms

def pca_cosine_similarity(input_histogram, histograms, pca, top_n=5):
    hist_values = np.array(list(histograms.values()))
    
    # PCA transform the histograms
    pca_histograms = pca.transform(hist_values)
    
    # PCA transform the input histogram
    pca_input_histogram = pca.transform([input_histogram])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(pca_input_histogram, pca_histograms)[0]
    
    # Get top N similar images
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    return [list(histograms.keys())[i] for i in top_indices]

def display_images(image_groups):
    fig, axes = plt.subplots(len(image_groups), 6, figsize=(18, len(image_groups) * 5))
    for i, group in enumerate(image_groups):
        for j, image_path in enumerate(group):
            if image_path is not None:
                try:
                    img = Image.open(image_path)
                    axes[i, j].imshow(img)
                    axes[i, j].axis("off")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    plt.show()

def get_image_path_from_db(image_id, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM images WHERE id=?", (image_id,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return None

def main():
    start_time = time.time()

    input_folder = 'examples/'
    pickle_file = "histograms.pkl"

    histograms = load_histograms(pickle_file)
    input_image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

    if len(input_image_paths) != 5:
        raise ValueError("Please ensure there are exactly 5 input images in the folder.")

    print(f"Input images: {input_image_paths}")

    conn = sqlite3.connect("images.db")

    all_similar_image_groups = []

    print("Extracting histograms...")
    with Pool(cpu_count()) as pool:
        input_histograms = list(tqdm(pool.imap(extract_histogram, input_image_paths), total=len(input_image_paths), desc="Extracting histograms"))

    # Fit PCA on the dataset histograms
    print("Running PCA...")
    hist_values = np.array(list(histograms.values()))
    pca = PCA(n_components=170)
    pca.fit(hist_values)
    print("PCA completed.")

    # Find similar images for each input image
    for input_image_path, input_histogram in zip(input_image_paths, input_histograms):
        similar_images = pca_cosine_similarity(input_histogram, histograms, pca)
        print(f"Similar images for {input_image_path}: {similar_images}")

        similar_image_paths = [input_image_path]
        for img_id in similar_images:
            similar_image_path = get_image_path_from_db(img_id, conn)
            if similar_image_path is not None:
                similar_image_paths.append(similar_image_path)
            else:
                print(f"Image path not found in database for ID {img_id}")

        all_similar_image_groups.append(similar_image_paths)

    conn.close()

    if all_similar_image_groups:
        print("Displaying images...")
        display_images(all_similar_image_groups)
    else:
        print("No similar images found.")

    end_time = time.time()
    duration = end_time - start_time
    print(f"The computation and display of images took {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
