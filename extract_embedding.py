import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import pickle
import time
import psutil  # To monitor memory usage
from tqdm import tqdm  # Progress bar
from torch.utils.data import Dataset, DataLoader
import uuid

print(torch.__version__)

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(
    *list(model.children())[:-1]
)  # Remove the classification layer
model = model.to(device)  # Move model to GPU
model.eval()

# Define the image transformation
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None


# Function to collect image paths
def collect_image_paths(root_dir, supported_formats=(".jpg", ".jpeg", ".png")):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(supported_formats):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths


# Function to extract embeddings and save checkpoints
def extract_embeddings(
    image_paths, batch_size, checkpoint_interval, output_dir="embeddings"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = ImageDataset(image_paths, transform=preprocess)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    embeddings_with_ids = []

    # Start timer
    start_time = time.time()

    for batch_images, batch_paths in tqdm(dataloader, desc="Overall Progress"):
        batch_images = batch_images.to(device)  # Move images to GPU
        with torch.no_grad():
            batch_embeddings = (
                model(batch_images).squeeze().cpu().numpy()
            )  # Ensure embeddings are moved back to CPU

        # Assign unique IDs and combine with image paths
        for embedding, img_path in zip(batch_embeddings, batch_paths):
            unique_id = str(uuid.uuid4())
            embeddings_with_ids.append((unique_id, embedding, img_path))

        print(f"Processed batch with {batch_images.shape[0]} images")

        # Clear GPU memory
        del batch_images
        torch.cuda.empty_cache()

        # Save checkpoint
        if len(embeddings_with_ids) % checkpoint_interval < batch_size:
            checkpoint_time = time.time()
            checkpoint_path = os.path.join(
                output_dir,
                f"embeddings_checkpoint_{len(embeddings_with_ids) // checkpoint_interval}.pkl",
            )
            with open(checkpoint_path, "wb") as f:
                pickle.dump(embeddings_with_ids, f)
            checkpoint_end_time = time.time()
            print(f"Checkpoint saved: {checkpoint_path}")
            print(
                f"Time spent on checkpointing: {checkpoint_end_time - checkpoint_time:.2f} seconds"
            )

    # Save final embeddings
    final_path = os.path.join(output_dir, "embeddings_with_ids.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(embeddings_with_ids, f)
    print(f"Final embeddings saved: {final_path}")

    # End timer
    end_time = time.time()

    # Calculate and print the total time taken
    total_time = end_time - start_time
    print(f"Total time taken for embedding extraction: {total_time:.2f} seconds")


if __name__ == "__main__":
    # Set data folder
    root_dir = "F:/data/image_data/"
    image_paths = collect_image_paths(root_dir)
    print(f"Number of images collected: {len(image_paths)}")

    # Set image batch size and checkpoint_interval (450,000 pictures)
    extract_embeddings(image_paths, batch_size=512, checkpoint_interval=100000)
