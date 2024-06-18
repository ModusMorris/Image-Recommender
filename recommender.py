import pickle
import sqlite3
from generator import load_image
from extract_rgb import extract_rgb_profile
from read_histogram import extract_histogram
from inception import extract_embedding
from similarity import compute_rgb_similarity, compute_histogram_similarity, compute_embedding_similarity

class ImageRecommender:
    def __init__(self, precomputed_data_path, db_path):
        # Load precomputed RGB profiles
        with open(precomputed_data_path, 'rb') as f:
            self.rgb_profiles = pickle.load(f)

        # Connect to SQLite database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.image_ids = list(self.rgb_profiles.keys())
    
    # def fetch_histogram(self, image_path):
    #     self.cursor.execute("SELECT histogram FROM images WHERE file_path=?", (image_path,))
    #     row = self.cursor.fetchone()
    #     if row:
    #         return pickle.loads(row[0])  # Assuming histogram is stored as pickled blob in the database
    #     else:
    #         raise KeyError(f"Histogram not found for image: {image_path}")

    # def fetch_embedding(self, image_path):
    #     self.cursor.execute("SELECT embedding FROM images WHERE file_path=?", (image_path,))
    #     row = self.cursor.fetchone()
    #     if row:
    #         return pickle.loads(row[0])  # Assuming embedding is stored as pickled blob in the database
    #     else:
    #         raise KeyError(f"Embedding not found for image: {image_path}")

    def recommend_similar_images(self, input_image_path, top_n=10):
        input_image = load_image(input_image_path)
        input_rgb = extract_rgb_profile(input_image)
        input_hist = extract_histogram(input_image)
        input_emb = extract_embedding(input_image)

        rgb_similarities = []
        hist_similarities = []
        emb_similarities = []

        for img_id in self.image_ids:
            rgb_sim = compute_rgb_similarity(input_rgb, self.rgb_profiles[img_id])
            hist_sim = compute_histogram_similarity(input_hist, self.fetch_histogram(img_id))
            emb_sim = compute_embedding_similarity(input_emb, self.fetch_embedding(img_id))

            rgb_similarities.append((img_id, rgb_sim))
            hist_similarities.append((img_id, hist_sim))
            emb_similarities.append((img_id, emb_sim))

        rgb_similarities.sort(key=lambda x: x[1])
        hist_similarities.sort(key=lambda x: x[1])
        emb_similarities.sort(key=lambda x: x[1], reverse=True)  # Cosine similarity: higher is better

        return {
            'rgb': rgb_similarities[:top_n],
            'histogram': hist_similarities[:top_n],
            'embedding': emb_similarities[:top_n]
        }

# Example usage
if __name__ == "__main__":
    import os

    # Correct the path to your image directory and input image
    image_directory = 'D:/Image-Recommender/archive'  # Replace with the path to your images
    input_image_path = 'D:/Image-Recommender/input_images'  # Replace with the path to the input image

    # Ensure the directory and file paths are correct and accessible
    if not os.path.exists(input_image_path):
        print(f"Input image path does not exist: {input_image_path}")
    else:
        # Ensure the directory and file paths are correct and accessible
        precomputed_data_path = 'D:/Image-Recommender/histograms.pkl'  # Replace with the path to your .pkl file
        db_path = 'D:/Image-Recommender/images.db'  # Path to the SQLite database

        if not os.path.exists(precomputed_data_path):
            print(f"Precomputed data path does not exist: {precomputed_data_path}")
        elif not os.path.exists(db_path):
            print(f"Database path does not exist: {db_path}")
        else:
            recommender = ImageRecommender(precomputed_data_path, db_path)
            similar_images = recommender.recommend_similar_images(input_image_path)

            print("Top 10 similar images based on RGB profile:")
            for img_id, sim_score in similar_images['rgb']:
                print(f"Image: {img_id}, Similarity Score: {sim_score}")

            print("\nTop 10 similar images based on Histogram:")
            for img_id, sim_score in similar_images['histogram']:
                print(f"Image: {img_id}, Similarity Score: {sim_score}")

            print("\nTop 10 similar images based on Embedding:")
            for img_id, sim_score in similar_images['embedding']:
                print(f"Image: {img_id}, Similarity Score: {sim_score}")
