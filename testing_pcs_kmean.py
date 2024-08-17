import pickle
import cupy as cp  # GPU-accelerated numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # Scikit-learn KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm  # Progress bar
from PIL import Image
import numpy as np

def load_single_pkl(file_path):
    with open(file_path, 'rb') as f:
        embeddings_with_ids = pickle.load(f)
        
    # Debugging: Number of embeddings loaded
    total_embeddings = len(embeddings_with_ids)
    print(f"Total embeddings loaded: {total_embeddings}")

    # Progress bar
    embeddings_with_progress = []
    for embedding_tuple in tqdm(embeddings_with_ids, desc="Loading embeddings", unit=" embedding"):
        embeddings_with_progress.append(embedding_tuple)

    return embeddings_with_progress

# Function to display images in a grid
def display_images(images, titles, rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()
    
    for img, title, ax in zip(images, titles, axes):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to get images near centroids
def get_images_near_centroids(cluster_labels, centroids, embeddings, image_paths, num_images=5):
    images = []
    titles = []
    for i, centroid in enumerate(centroids):
        # Get indices of images in the current cluster
        cluster_indices = np.where(cluster_labels == i)[0]
        
        # Check if the cluster has enough images
        if len(cluster_indices) == 0:
            continue
        
        # Calculate distances to the centroid within the cluster
        distances = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
        
        # Get indices of the closest images within the cluster
        closest_indices = cluster_indices[np.argsort(distances)[:min(num_images, len(cluster_indices))]]
        
        # Load and store the images
        for idx in closest_indices:
            img = Image.open(image_paths[idx])
            images.append(img)
            titles.append(f"Cluster {i}, Img {idx}")
    
    return images, titles

# Load the embeddings
file_path = 'pkl_files/embeddings_with_ids.pkl'  # Replace with the actual path to your pkl file
embeddings_with_ids = load_single_pkl(file_path)

# Convert embeddings to a CuPy array for GPU processing
embeddings = cp.array([item[1] for item in embeddings_with_ids])
image_paths = [item[2] for item in embeddings_with_ids]  # Optional: might be useful for labeling

# Step 1: Reduce dimensions with PCA
print("Starting PCA dimensionality reduction...")
pca = PCA(n_components=2)  # Reduce directly to 2D
pca_embeddings = pca.fit_transform(embeddings.get())  # Convert to NumPy for PCA

# Step 2: Perform KMeans clustering on the 2D PCA result
n_clusters = 15  # Adjust based on your dataset and needs
print(f"Starting KMeans clustering with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(pca_embeddings)

# Calculate cluster centroids
centroids = kmeans.cluster_centers_

# Step 3: Plot with labels
print("Generating plot...")
plt.figure(figsize=(12, 8))
plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=cluster_labels, cmap='Spectral', s=1)
plt.colorbar()

for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], str(i), fontsize=9, weight='bold', color='black')

plt.title('2D Visualization of Image Communities with Labels (PCA + KMeans)')
plt.show()

# Step 4: Display images near centroids
print("Displaying images near centroids...")
images, titles = get_images_near_centroids(cluster_labels, centroids, pca_embeddings, image_paths)

# Display the images in a grid
display_images(images, titles, rows=3, cols=5)

print("Process completed successfully!")
