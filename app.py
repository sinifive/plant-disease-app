from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model and class names paths
MODEL_PATH = "model_output/model_scripted.pth"
CLASS_NAMES_PATH = "model_output/class_names.txt"

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and class_names variables
model = None
class_names = {}

# Try to load the model
try:
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Try to load class names
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        for line in f:
            index, name = line.strip().split(':', 1)
            class_names[int(index)] = name
    print(f"Loaded {len(class_names)} class names")
except Exception as e:
    print(f"Error loading class names: {str(e)}")

# Dictionary with disease information (you should expand this with your own data)
disease_info = {
    "Tomato_Late_blight": {
        "description": "Late blight is a potentially devastating disease caused by the fungus Phytophthora infestans. It spreads quickly in wet and humid conditions.",
        "treatments": [
            "Remove and destroy infected plant parts",
            "Apply copper-based fungicide",
            "Improve air circulation around plants",
            "Water at the base of plants to avoid leaf wetness",
            "Consider resistant varieties for future planting"
        ]
    },
    # Add more diseases here
}

# Default info for diseases not in the dictionary
default_info = {
    "description": "A plant disease that affects plant health and yield.",
    "treatments": [
        "Remove infected plant parts",
        "Apply appropriate fungicide or pesticide",
        "Ensure proper plant spacing for good air circulation",
        "Maintain good garden hygiene",
        "Contact a local extension office for specific guidance"
    ]
}

# Image transformation for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read the image
        image_file = request.files['image']
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        # Preprocess the image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_class_index].item() * 100)
        
        # Get the class name
        if predicted_class_index in class_names:
            disease_name = class_names[predicted_class_index].replace('_', ' ')
        else:
            disease_name = f"Unknown (Class {predicted_class_index})"
        
        # Get disease information
        disease_key = class_names.get(predicted_class_index, "Unknown")
        info = disease_info.get(disease_key, default_info)
        
        result = {
            'disease': disease_name,
            'confidence': confidence,
            'description': info['description'],
            'treatments': info['treatments']
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

print("Starting Plant Disease Classification API")
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
