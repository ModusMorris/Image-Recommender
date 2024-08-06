import pickle
import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock  # For Manhattan distance
from torchvision import models, transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Load the embeddings with unique IDs and image paths
start_time = time.time()
with open('embeddings/embeddings_with_ids.pkl', 'rb') as f:
    embeddings_with_ids = pickle.load(f)
load_time = time.time() - start_time
print(f"Time to load embeddings: {load_time:.2f} seconds")

# Separate unique IDs, embeddings, and image paths
unique_ids, embeddings, image_paths = zip(*embeddings_with_ids)
embeddings = np.array(embeddings)
print("Embeddings loaded and converted to numpy array.")

# Define the image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_path):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()  # Move back to CPU and get numpy array
    return embedding

print("Model defined and image preprocessing set.")

def calculate_similarities(input_embedding, embeddings):
    cosine_sim = cosine_similarity([input_embedding], embeddings)[0]
    euclidean_sim = -euclidean_distances([input_embedding], embeddings)[0]  # Negate to make it a similarity
    manhattan_sim = -np.array([cityblock(input_embedding, emb) for emb in embeddings])  # Negate to make it a similarity
    return cosine_sim, euclidean_sim, manhattan_sim

print("Similarity functions defined.")

# Load the input image
input_image_path = 'input_images/new_try3.jpg'  # Adjust the path as needed
print(f"Input image path: {input_image_path}")

input_embedding = get_image_embedding(input_image_path)
print("Input image embedding calculated.")

# Calculate similarities
cosine_sim, euclidean_sim, manhattan_sim = calculate_similarities(input_embedding, embeddings)
print("Similarities calculated.")

# Get top 5 indices for each similarity measure
top_indices_cosine = cosine_sim.argsort()[-5:][::-1]
top_indices_euclidean = euclidean_sim.argsort()[-5:][::-1]
top_indices_manhattan = manhattan_sim.argsort()[-5:][::-1]

print("Top indices for each similarity measure found.")

# Combine top indices (ensuring uniqueness) and sort by cosine similarity for display
top_indices = np.unique(np.concatenate((top_indices_cosine, top_indices_euclidean, top_indices_manhattan)))
top_indices = top_indices[np.argsort(cosine_sim[top_indices])[::-1]]

top_images = [image_paths[idx] for idx in top_indices[:5]]
top_similarities = cosine_sim[top_indices[:5]]

print("Top images and similarities selected.")

def plot_images(input_image_path, top_images, similarity_percentages):
    fig, axes = plt.subplots(2, 5, figsize=(17, 9))  # Create a 2x5 grid
    
    # Plot the input image
    input_img = Image.open(input_image_path)
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')
    
    # Leave the remaining columns of the first row empty
    for j in range(1, 5):
        axes[0, j].axis('off')
    
    # Plot the top 5 similar images in the second row
    for i, (image_path, similarity) in enumerate(zip(top_images, similarity_percentages)):
        if os.path.exists(image_path):
            img = Image.open(image_path)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Similarity: {similarity * 100:.2f}%\n{os.path.basename(image_path)}")
            axes[1, i].axis('off')
        else:
            axes[1, i].set_title("Image not found")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_images(input_image_path, top_images, top_similarities)
print("Images plotted.")