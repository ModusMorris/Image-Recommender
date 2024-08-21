import os
import pickle
import sqlite3
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
from multiprocessing import cpu_count

# Global Cache für bereits berechnete Histogramme
histogram_cache = {}

@njit(parallel=True)
def intersection_similarity_vectorized(input_histogram, hist_list):
    similarities = np.zeros(hist_list.shape[0], dtype=np.float32)
    for i in prange(hist_list.shape[0]):
        similarities[i] = np.sum(np.minimum(input_histogram, hist_list[i]))
    return similarities

def extract_histogram(image_path):
    if (cached_hist := histogram_cache.get(image_path)) is not None:
        return cached_hist

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((128, 128))  # Reduzierte Bildgröße für schnellere Verarbeitung
        histogram = np.array(img.histogram(), dtype=np.float32).reshape((3, 256))
        histogram /= histogram.sum()
        histogram_cache[image_path] = histogram.flatten()

    return histogram_cache[image_path]

def load_histograms(pickle_file):
    with open(pickle_file, "rb") as f:
        histograms = pickle.load(f)
    # Konvertiere alle Histogramme zu float32 für geringeren Speicherverbrauch
    for k, v in histograms.items():
        histograms[k] = np.array(v, dtype=np.float32)
    return histograms

def find_similar_images_for_one_input(input_histogram, histograms, top_n=5):
    hist_list = np.array(list(histograms.values()), dtype=np.float32)
    similarities = intersection_similarity_vectorized(input_histogram, hist_list)
    sorted_indices = np.argsort(-similarities)[:top_n]  # Sort in descending order
    return [list(histograms.keys())[i] for i in sorted_indices], similarities[sorted_indices]

def find_aggregated_similar_images(input_histograms, histograms, top_n=5):
    hist_list = np.array(list(histograms.values()), dtype=np.float32)
    aggregated_similarities = np.zeros(hist_list.shape[0], dtype=np.float32)

    for input_hist in input_histograms:
        similarities = intersection_similarity_vectorized(input_hist, hist_list)
        aggregated_similarities += similarities

    aggregated_similarities /= len(input_histograms)
    sorted_indices = np.argsort(-aggregated_similarities)[:top_n]  # Sort in descending order
    return [list(histograms.keys())[i] for i in sorted_indices], aggregated_similarities[sorted_indices]

def display_images(image_groups, titles):
    num_images = len(image_groups[0])
    num_rows = len(image_groups)
    fig, axes = plt.subplots(num_rows, num_images, figsize=(18, num_rows * 5))

    for i, (group, title_group) in enumerate(zip(image_groups, titles)):
        for j, (image_path, title) in enumerate(zip(group, title_group)):
            if image_path is not None:
                try:
                    img = Image.open(image_path)
                    axes[i, j].imshow(img)
                    axes[i, j].axis("off")
                    axes[i, j].set_title(title, fontsize=12)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    axes[i, j].axis("off")
                    axes[i, j].text(0.5, 0.5, "Error", ha="center", va="center")

    plt.tight_layout()
    plt.show()

def get_image_paths_from_db(image_ids, conn):
    cursor = conn.cursor()
    query = "SELECT id, file_path FROM images WHERE id IN ({seq})".format(
        seq=",".join(["?"] * len(image_ids))
    )
    cursor.execute(query, image_ids)
    result = cursor.fetchall()
    return {row[0]: row[1] for row in result}

def main():
    start_time = time.time()

    input_folder = "examples/"
    pickle_file = "histograms.pkl"

    histograms = load_histograms(pickle_file)
    input_image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(input_image_paths) != 5:
        raise ValueError(
            "Please ensure there are exactly 5 input images in the folder."
        )

    print(f"Input images: {input_image_paths}")

    conn = sqlite3.connect("images.db")

    all_similar_image_groups = []
    all_titles = []

    extract_hist_start = time.time()
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        input_histograms = list(executor.map(extract_histogram, input_image_paths))
    extract_hist_end = time.time()
    print(
        f"Time to extract histograms: {extract_hist_end - extract_hist_start:.2f} seconds"
    )

    find_sim_start = time.time()
    for input_image_path, input_histogram in zip(input_image_paths, input_histograms):
        similar_images, similarities = find_similar_images_for_one_input(
            input_histogram, histograms
        )
        similar_image_paths = [input_image_path]
        titles = ["Input Image"]
        image_paths = get_image_paths_from_db(similar_images, conn)

        for img_id, similarity in zip(similar_images, similarities):
            if img_id in image_paths:
                similar_image_paths.append(image_paths[img_id])
                titles.append(f"Similarity: {similarity*100:.2f}%")
            else:
                print(f"Image path not found in database for ID {img_id}")

        all_similar_image_groups.append(similar_image_paths)
        all_titles.append(titles)
    find_sim_end = time.time()
    print(
        f"Time to find all similar images: {find_sim_end - find_sim_start:.2f} seconds"
    )

    agg_sim_start = time.time()
    aggregated_similar_images, aggregated_similarities = find_aggregated_similar_images(
        input_histograms, histograms
    )
    aggregated_similar_image_paths = []
    aggregated_titles = []
    image_paths = get_image_paths_from_db(aggregated_similar_images, conn)

    for img_id, similarity in zip(aggregated_similar_images, aggregated_similarities):
        if img_id in image_paths:
            aggregated_similar_image_paths.append(image_paths[img_id])
            aggregated_titles.append(f"Mean Sim: {similarity*100:.2f}%")
        else:
            print(f"Image path not found in database for ID {img_id}")

    all_similar_image_groups.append(aggregated_similar_image_paths)
    all_titles.append(aggregated_titles)
    agg_sim_end = time.time()
    print(
        f"Time to find aggregated similar images: {agg_sim_end - agg_sim_start:.2f} seconds"
    )

    conn.close()

    end_time = time.time()
    duration = end_time - start_time
    print(f"The computation and display of images took {duration:.2f} seconds.")

    plot_start = time.time()
    display_images(all_similar_image_groups, all_titles)
    plot_end = time.time()
    print(f"Time to display images: {plot_end - plot_start:.2f} seconds")




if __name__ == "__main__":
    main()
