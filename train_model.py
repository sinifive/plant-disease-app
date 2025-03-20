import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import time

def create_model(num_classes):
    """Create a PyTorch model for plant disease classification"""
    # Use a pre-trained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Freeze all layers except the last few
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(dataset_path, output_path, epochs=10, batch_size=32):
    """Train the plant disease classification model"""
    print(f"Starting model training with dataset: {dataset_path}")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load datasets
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    # Get class names
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Save class names
    with open(os.path.join(output_path, 'class_names.txt'), 'w') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{i}:{class_name}\n")
    
    # Create model
    model = create_model(num_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(output_path, 'best_model.pth'))
                print(f"Saved new best model with accuracy: {best_acc:.4f}")
    
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    
    # Create scripted model for inference
    model.load_state_dict(torch.load(os.path.join(output_path, 'best_model.pth')))
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    example = example.to(device)
    scripted_model = torch.jit.trace(model, example)
    scripted_model.save(os.path.join(output_path, 'model_scripted.pth'))
    print(f"Saved scripted model to {os.path.join(output_path, 'model_scripted.pt')}")
    
    return model, class_names

if __name__ == "__main__":
    dataset_path = "plant_disease_dataset"  # Should contain 'train' and 'val' subdirectories
    output_path = "model_output"
    
    model, class_names = train_model(dataset_path, output_path)
    
    print("Training complete!")
    print(f"Model saved to {output_path}")
    print(f"Classes: {class_names}")
