import os
import pickle
import sqlite3
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count

def extract_histogram(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        histogram = np.array(img.histogram()).reshape((3, 256)).astype(float)
        histogram /= histogram.sum()
        return histogram.flatten()

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

def load_histograms(pickle_file):
    with open(pickle_file, "rb") as f:
        histograms = pickle.load(f)
    return histograms

def find_similar_images_for_one_input(input_histogram, histograms, top_n=5):
    hist_list = np.array(list(histograms.values()))
    distances = np.array([chi2_distance(input_histogram, hist) for hist in hist_list])
    sorted_indices = np.argsort(distances)[:top_n]
    return [list(histograms.keys())[i] for i in sorted_indices]

def find_aggregated_similar_images(input_histograms, histograms, top_n=5):
    aggregated_distances = np.zeros(len(histograms))
    hist_list = np.array(list(histograms.values()))
    
    for input_hist in input_histograms:
        distances = np.array([chi2_distance(input_hist, hist) for hist in hist_list])
        aggregated_distances += distances

    sorted_indices = np.argsort(aggregated_distances)[:top_n]
    return [list(histograms.keys())[i] for i in sorted_indices]

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

    with Pool(cpu_count()) as pool:
        input_histograms = pool.map(extract_histogram, input_image_paths)

    # Find individually similar images
    for input_image_path, input_histogram in zip(input_image_paths, input_histograms):
        similar_images = find_similar_images_for_one_input(input_histogram, histograms)
        print(f"Similar images for {input_image_path}: {similar_images}")

        similar_image_paths = [input_image_path]
        for img_id in similar_images:
            similar_image_path = get_image_path_from_db(img_id, conn)
            if similar_image_path is not None:
                similar_image_paths.append(similar_image_path)
            else:
                print(f"Image path not found in database for ID {img_id}")

        print(f"Similar images for {input_image_path}:")
        for path in similar_image_paths:
            print(path)

        all_similar_image_groups.append(similar_image_paths)

    # Find aggregated similar images
    aggregated_similar_images = find_aggregated_similar_images(input_histograms, histograms)
    aggregated_similar_image_paths = []
    for img_id in aggregated_similar_images:
        similar_image_path = get_image_path_from_db(img_id, conn)
        if similar_image_path is not None:
            aggregated_similar_image_paths.append(similar_image_path)
        else:
            print(f"Image path not found in database for ID {img_id}")

    all_similar_image_groups.append(aggregated_similar_image_paths)

    conn.close()
    display_images(all_similar_image_groups)

    end_time = time.time()
    duration = end_time - start_time
    print(f"The computation and display of images took {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
