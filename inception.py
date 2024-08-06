import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3

# Load the InceptionV3 model
model = inception_v3(pretrained=True)
model.fc = torch.nn.Identity()  # Remove the final classification layer
model.eval()  # Set the model to evaluation mode

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_embedding(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image)
    return embedding.flatten().numpy()
