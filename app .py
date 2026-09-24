
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# Define the SimpleCNN model architecture (must be identical to training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Streamlit App --- 

st.title("Image Classification App")
st.write("Upload an image to classify it as Plastic Bottle or Steel Bottle.")

# Hardcoded class names (must be in the same order as during training)
class_names = ['Plastic Bottle', 'Steel Bottle']

# Load the trained model
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=len(class_names)) # Ensure num_classes matches your training
    # !!! IMPORTANT: Replace 'model_state_dict.pth' with the actual path to your saved model weights !!!
    # You need to save your trained model's state_dict like this: 
    # torch.save(model.state_dict(), 'model_state_dict.pth') after training.
    model_path = 'model_state_dict.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"Model weights not found at {model_path}. Please train and save the model first.")
        return None

model = load_model()

# Image transformations for inference (must be identical to testing transformations)
preprocess = transforms.Compose([
    transforms.Resize((250, 250)), # Resize to the target size used during training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")

        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)

        # Get probabilities and predicted class
        probabilities = torch.softmax(output, dim=1)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = probabilities[0][predicted_idx.item()].item()

        st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
    else:
        st.warning("Model is not loaded. Please ensure 'model_state_dict.pth' exists.")
