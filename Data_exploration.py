import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load the embeddings from the pkl file
with open('pkl_files/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

print(f"Loaded {len(embeddings)} embeddings.")

# Step 2: Reduce dimensionality using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

# Step 3: Apply K-means clustering
n_clusters = 30  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced_embeddings)
print(f"Cluster labels: {clusters}")

# Step 4: Plot the clustered embeddings
plt.figure(figsize=(12, 8))
for i in range(n_clusters):
    plt.scatter(reduced_embeddings[clusters == i, 0], reduced_embeddings[clusters == i, 1], label=f"Cluster {i}")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of Image Embeddings with K-means Clusters")
plt.legend()
plt.show()
