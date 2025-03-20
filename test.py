import torch
import os
from torchvision import models
from torchvision.models import ResNet18_Weights

# Paths
output_path = "model_output"
best_model_path = os.path.join(output_path, "best_model.pth")
scripted_model_path = os.path.join(output_path, "model_scripted.pth")

# Load checkpoint to get correct num_classes
checkpoint = torch.load(best_model_path, map_location="cpu")
num_classes = checkpoint["fc.bias"].shape[0]  # Dynamically determine class count
print(f"Detected {num_classes} classes from checkpoint.")

# Define the model function
def create_model(num_classes):
    """Create a PyTorch model for plant disease classification"""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Freeze all layers except the last few
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    
    return model

# Create model with correct class count
model = create_model(num_classes)

# Load trained weights
model.load_state_dict(checkpoint)

# Set to evaluation mode
model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(model)

# Save scripted model
scripted_model.save(scripted_model_path)

print(f"✅ Model successfully scripted and saved to: {scripted_model_path}")
