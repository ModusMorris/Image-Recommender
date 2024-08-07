import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, Dataset

# Pfad zu den Bildern
image_dir = r"J:\data"  # Verwenden Sie rohe Strings, um Escape-Sequenzen zu vermeiden
checkpoint_file = "image_features_checkpoint.pkl"
final_file = "image_features.pkl"

# Gerät auf GPU setzen, falls verfügbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Laden des vortrainierten ResNet-Modells
model = models.resnet50(weights="IMAGENET1K_V1")
model = nn.Sequential(*list(model.children())[:-1])  # Entfernen des letzten FC-Layers
model = model.to(device)
model.eval()

# Transformationen für die Eingabebilder
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return img_path, image


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)
    return {}


def save_checkpoint(feature_dict, checkpoint_file):
    with open(checkpoint_file, "wb") as f:
        pickle.dump(feature_dict, f)


def collect_image_paths(image_dir):
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(("jpg", "jpeg", "png")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def extract_features_in_batches(image_dir, batch_size=32, checkpoint_interval=1000):
    image_paths = collect_image_paths(image_dir)
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    feature_dict = load_checkpoint(checkpoint_file)
    processed_images = set(feature_dict.keys())
    total_images = len(image_paths)
    processed_count = len(processed_images)

    with tqdm(
        total=total_images, initial=processed_count, desc="Processing images"
    ) as pbar:
        for img_paths, images in dataloader:
            images = images.to(device)
            with torch.no_grad():
                features = model(images)
            features = features.cpu().numpy()
            for img_path, feature in zip(img_paths, features):
                if img_path not in processed_images:
                    feature_dict[img_path] = feature.flatten()
                    processed_images.add(img_path)
                    processed_count += 1
                    pbar.update(1)

            # Speichern des Checkpoints nach jeder bestimmten Anzahl von Bildern
            if processed_count % checkpoint_interval == 0:
                save_checkpoint(feature_dict, checkpoint_file)

    return feature_dict


if __name__ == "__main__":
    # Feature-Vektoren in Batches extrahieren und speichern
    feature_dict = extract_features_in_batches(image_dir)

    # Speichern in einer endgültigen Pickle-Datei
    with open(final_file, "wb") as f:
        pickle.dump(feature_dict, f)

    # Checkpoint-Datei entfernen, nachdem der Prozess abgeschlossen ist
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
