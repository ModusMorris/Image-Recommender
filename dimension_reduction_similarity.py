import os
import pickle
import sqlite3
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import cpu_count

# Global Cache für bereits berechnete Histogramme
histogram_cache = {}
pca_cache = None  # Cache für PCA-Ergebnisse

def extract_histogram(image_path):
    if image_path in histogram_cache:
        return histogram_cache[image_path]

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

def pca_cosine_similarity(input_histogram, histograms, pca, top_n=5):
    global pca_cache
    if pca_cache is None:
        hist_values = np.array(list(histograms.values()), dtype=np.float32)
        pca_histograms = pca.transform(hist_values)
        pca_cache = pca_histograms
    else:
        pca_histograms = pca_cache

    pca_input_histogram = pca.transform([input_histogram])
    similarities = cosine_similarity(pca_input_histogram, pca_histograms)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(list(histograms.keys())[i], similarities[i]) for i in top_indices]

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

def get_image_path_from_db(image_id, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM images WHERE id=?", (image_id,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return None

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

    # Timing: Extract histograms
    extract_hist_start = time.time()
    with ProcessPoolExecutor(max_workers=min(cpu_count(), len(input_image_paths))) as executor:
        input_histograms = list(
            tqdm(
                executor.map(extract_histogram, input_image_paths),
                total=len(input_image_paths),
                desc="Extracting histograms",
            )
        )
    extract_hist_end = time.time()
    print(
        f"Time to extract histograms: {extract_hist_end - extract_hist_start:.2f} seconds"
    )

    # Calculate average histogram
    average_histogram = np.mean(input_histograms, axis=0)

    # Timing: PCA fitting
    pca_start = time.time()
    hist_values = np.array(list(histograms.values()), dtype=np.float32)
    pca = PCA(n_components=170)
    pca.fit(hist_values)
    pca_end = time.time()
    print(f"Time to fit PCA: {pca_end - pca_start:.2f} seconds")

    # Timing: Find similar images
    find_sim_start = time.time()
    with ThreadPoolExecutor(max_workers=min(cpu_count(), len(input_image_paths))) as executor:
        similar_images_results = list(
            executor.map(
                lambda hist: pca_cosine_similarity(hist, histograms, pca),
                input_histograms
            )
        )
    for input_image_path, similar_images in zip(input_image_paths, similar_images_results):
        similar_image_paths = [(input_image_path, 1.0)]
        titles = ["Input Image"]
        for img_id, similarity in similar_images:
            similar_image_path = get_image_path_from_db(img_id, conn)
            if similar_image_path is not None:
                similar_image_paths.append((similar_image_path, similarity))
                titles.append(f"Similarity: {similarity*100:.2f}%")
            else:
                print(f"Image path not found in database for ID {img_id}")

        all_similar_image_groups.append(similar_image_paths)
        all_titles.append(titles)
    find_sim_end = time.time()
    print(
        f"Time to find all similar images: {find_sim_end - find_sim_start:.2f} seconds"
    )

    # Timing: Find aggregated similar images
    agg_sim_start = time.time()
    overall_similar_images = pca_cosine_similarity(average_histogram, histograms, pca)
    overall_similar_image_paths = []
    overall_titles = []
    for img_id, similarity in overall_similar_images:
        similar_image_path = get_image_path_from_db(img_id, conn)
        if similar_image_path is not None:
            overall_similar_image_paths.append((similar_image_path, similarity))
            overall_titles.append(f"Mean Sim: {similarity*100:.2f}%")
        else:
            print(f"Image path not found in database for ID {img_id}")

    all_similar_image_groups.append(overall_similar_image_paths)
    all_titles.append(overall_titles)
    agg_sim_end = time.time()
    print(
        f"Time to find aggregated similar images: {agg_sim_end - agg_sim_start:.2f} seconds"
    )

    conn.close()

    # Timing: Display images
    plot_start = time.time()
    # Extract paths from tuples for display
    all_similar_image_groups_paths = [
        [img_path for img_path, similarity in group]
        for group in all_similar_image_groups
    ]
    display_images(all_similar_image_groups_paths, all_titles)
    plot_end = time.time()
    print(f"Time to display images: {plot_end - plot_start:.2f} seconds")

    end_time = time.time()
    duration = end_time - start_time
    print(f"The computation and display of images took {duration:.2f} seconds.")


if __name__ == "__main__":
    main()
