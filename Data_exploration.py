import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Step 1: Load the embeddings with IDs and image paths
with open('pkl_files/embeddings_with_ids.pkl', 'rb') as f:
    embeddings_with_ids = pickle.load(f)

# Extract the embeddings and image paths
embeddings = np.array([item[1] for item in embeddings_with_ids])
image_paths = [item[2] for item in embeddings_with_ids]

print(f"Loaded {len(embeddings)} embeddings and their corresponding image paths.")

# Step 2: Reduce dimensionality using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

# Step 3: Apply K-means clustering
n_clusters = 30  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced_embeddings)
print(f"Cluster labels: {clusters}")

# Step 4: Find the closest images to the centroids in the original space
centroids = kmeans.cluster_centers_

# Map centroids back to the original space using inverse transform (approximate)
centroids_original_space = pca.inverse_transform(centroids)

# Calculate pairwise distances between centroids and all embeddings
distances = pairwise_distances(centroids_original_space, embeddings)

# Step 5: Ensure unique closest images for each centroid
assigned_images = set()
closest_images = []

for i in range(n_clusters):
    sorted_indices = np.argsort(distances[i])
    for index in sorted_indices:
        if index not in assigned_images:
            closest_images.append(index)
            assigned_images.add(index)
            break

# Step 6: Plot the clustered embeddings
plt.figure(figsize=(16, 12))
for i in range(n_clusters):
    plt.scatter(reduced_embeddings[clusters == i, 0], reduced_embeddings[clusters == i, 1], label=f"Cluster {i}")

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

# Add images corresponding to the centroids
for i, index in enumerate(closest_images):
    img_path = image_paths[index]
    image = Image.open(img_path)
    
    # Resize the image to a smaller size if necessary
    max_size = 128  # Maximum size for either dimension of the image
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Increase the zoom level to make the images larger
    imagebox = OffsetImage(image, zoom=0.3)
    ab = AnnotationBbox(imagebox, (centroids[i, 0], centroids[i, 1]),
                        frameon=False, pad=0.5)
    plt.gca().add_artist(ab)
    plt.text(centroids[i, 0], centroids[i, 1], f'Centroid {i}', fontsize=9, weight='bold', ha='center', va='center')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of Image Embeddings with K-means Clusters and Centroid Images")
plt.legend()
plt.tight_layout()
plt.show()
