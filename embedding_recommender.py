import os
import pickle
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock  # For Manhattan distance
from torchvision import models, transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Load the embeddings with unique IDs and image paths
start_time = time.time()
with open("pkl_files/embeddings_with_ids.pkl", "rb") as f:
    embeddings_with_ids = pickle.load(f)
load_time = time.time() - start_time
print(f"Time to load embeddings: {load_time:.2f} seconds")

# Separate unique IDs, embeddings, and image paths
unique_ids, embeddings, image_paths = zip(*embeddings_with_ids)
embeddings = np.array(embeddings)
print("Embeddings loaded and converted to numpy array.")

# Define the image transformation
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_image_embedding(image_path):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(
        *list(model.children())[:-1]
    )  # Remove the classification layer
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img = Image.open(image_path).convert("RGB")
    img_tensor = (
        preprocess(img).unsqueeze(0).to(device)
    )  # Add batch dimension and move to GPU
    with torch.no_grad():
        embedding = (
            model(img_tensor).squeeze().cpu().numpy()
        )  # Move back to CPU and get numpy array
    return embedding


print("Model defined and image preprocessing set.")


def calculate_similarities(input_embedding, embeddings):
    cosine_sim = cosine_similarity([input_embedding], embeddings)[0]
    euclidean_sim = -euclidean_distances([input_embedding], embeddings)[
        0
    ]  # Negate to make it a similarity
    manhattan_sim = -np.array(
        [cityblock(input_embedding, emb) for emb in embeddings]
    )  # Negate to make it a similarity
    return cosine_sim, euclidean_sim, manhattan_sim


# Process all images in the /examples folder
example_folder = "examples"
example_images = [
    os.path.join(example_folder, f)
    for f in os.listdir(example_folder)
    if f.endswith(("png", "jpg", "jpeg"))
]

num_images = len(example_images)
fig, axes = plt.subplots(
    num_images + 1, 6, figsize=(20, 4 * (num_images + 1))
)  # Create a grid with num_images + 1 rows and 6 columns

# Store all cosine similarities
all_cosine_similarities = []

embedding_start_time = time.time()

for i, input_image_path in enumerate(example_images):
    # Get the embedding of the input image
    input_embedding_start = time.time()
    input_embedding = get_image_embedding(input_image_path)
    input_embedding_end = time.time()
    print(
        f"Time to compute embedding for {input_image_path}: {input_embedding_end - input_embedding_start:.2f} seconds"
    )
    print(f"Input image embedding calculated for {input_image_path}")

    # Calculate similarities
    similarities_start = time.time()
    cosine_sim, euclidean_sim, manhattan_sim = calculate_similarities(
        input_embedding, embeddings
    )
    similarities_end = time.time()
    print(
        f"Time to calculate similarities for {input_image_path}: {similarities_end - similarities_start:.2f} seconds"
    )
    print("Similarities calculated.")

    # Get top 5 indices for each similarity measure
    top_indices_cosine = cosine_sim.argsort()[-5:][::-1]
    top_indices_euclidean = euclidean_sim.argsort()[-5:][::-1]
    top_indices_manhattan = manhattan_sim.argsort()[-5:][::-1]

    print("Top indices for each similarity measure found.")

    # Combine top indices (ensuring uniqueness) and sort by cosine similarity for display
    top_indices = np.unique(
        np.concatenate(
            (top_indices_cosine, top_indices_euclidean, top_indices_manhattan)
        )
    )
    top_indices = top_indices[np.argsort(cosine_sim[top_indices])[::-1]]

    top_images = [image_paths[idx] for idx in top_indices[:5]]
    top_similarities = cosine_sim[top_indices[:5]]

    print(f"Top images and similarities selected for {input_image_path}")

    # Plot the input image in the first column of the current row
    input_img = Image.open(input_image_path)
    axes[i, 0].imshow(input_img)
    axes[i, 0].set_title("Input Image")
    axes[i, 0].axis("off")

    # Plot the top 5 similar images in the remaining columns of the current row
    for j, (image_path, similarity) in enumerate(zip(top_images, top_similarities)):
        if os.path.exists(image_path):
            img = Image.open(image_path)
            axes[i, j + 1].imshow(img)
            axes[i, j + 1].set_title(
                f"Similarity: {similarity * 100:.2f}%\n{os.path.basename(image_path)}"
            )
            axes[i, j + 1].axis("off")
        else:
            axes[i, j + 1].set_title("Image not found")
            axes[i, j + 1].axis("off")

    # Append the cosine similarities
    all_cosine_similarities.append(cosine_sim)

embedding_end_time = time.time()
print(
    f"Time to process all embeddings: {embedding_end_time - embedding_start_time:.2f} seconds"
)

# Calculate the mean cosine similarity across all input images
mean_cosine_similarities = np.mean(all_cosine_similarities, axis=0)
top_indices_combined = mean_cosine_similarities.argsort()[-5:][::-1]

top_images_combined = [image_paths[idx] for idx in top_indices_combined]
top_similarities_combined = mean_cosine_similarities[top_indices_combined]

# Plot the overall top 5 similar images in the last row
axes[-1, 0].set_title("Top 5 Similar Images Overall")
axes[-1, 0].axis("off")

for j, (image_path, similarity) in enumerate(
    zip(top_images_combined, top_similarities_combined)
):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        axes[-1, j + 1].imshow(img)
        axes[-1, j + 1].set_title(
            f"Similarity: {similarity * 100:.2f}%\n{os.path.basename(image_path)}"
        )
        axes[-1, j + 1].axis("off")
    else:
        axes[-1, j + 1].set_title("Image not found")
        axes[-1, j + 1].axis("off")

plot_start = time.time()
plt.tight_layout()
plt.show()
plot_end = time.time()
print(f"Time to display images: {plot_end - plot_start:.2f} seconds")

end_time = time.time()
duration = end_time - start_time
print(f"The computation and display of images took {duration:.2f} seconds.")
