### GestureNet - Classifier Static Hand Gestures (CPU - Model)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from src.torch_tests.conf import CFG_ASLNet, __DEVICE__

def imshow(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()  # Seleciona índice aleatório
        img_tensor, label = data[sample_idx]
        
        # Denormaliza para exibição
        img = img_tensor * 0.5 + 0.5  # Desfaz a normalização (-0.5 a 0.5 para 0 a 1)
        img = img.squeeze(0)  # Remove o canal único

        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()

def _plot_predictions(model, dataset, device, epoch, plot_dir, num_samples=6):
    """Plot sample predictions with labels"""
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices, 1):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
        
        # Denormalize image
        img = image.squeeze().cpu().numpy()
        img = img * 0.5 + 0.5  # Undo normalization
        
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {CFG_ASLNet.LABELS[label]}\nPred: {CFG_ASLNet.LABELS[pred.item()]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(plot_dir/f"predictions_epoch_{epoch}.png")
    plt.close()

def _plot_metrics(train_losses, val_accuracies, plot_dir):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_dir/"training_metrics.png")
    plt.close()

def _plot_confusion_matrix(preds, labels, plot_dir):
    """Plot confusion matrix at end of training"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=CFG_ASLNet.LABELS, 
                yticklabels=CFG_ASLNet.LABELS)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(plot_dir/"confusion_matrix.png")
    plt.close()

# Custom Dataset with error handling
class ASLDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('L')  # Grayscale
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return self[(idx + 1) % len(self)]  # Skip corrupted files

# Enhanced CNN Model
class ASLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (CFG_ASLNet.IMG_SIZE//8)**2, 512),
            nn.ReLU(),
            nn.Dropout(CFG_ASLNet.DROPOUT),
            nn.Linear(512, CFG_ASLNet.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Data preparation
def prepare_data():
    # Collect all labeled images
    data = []
    for label_idx, label in enumerate(CFG_ASLNet.LABELS):
        class_dir = os.path.join(CFG_ASLNet.DATA_DIR, label)
        if not os.path.exists(class_dir):
            continue
            
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                data.append((
                    os.path.join(class_dir, file),
                    label_idx
                ))

    # Split with stratification
    train_data, test_data = train_test_split(
        data, 
        test_size=0.2, 
        stratify=[d[1] for d in data],
        random_state=42
    )
    
    # Transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.Resize((CFG_ASLNet.IMG_SIZE, CFG_ASLNet.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((CFG_ASLNet.IMG_SIZE, CFG_ASLNet.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    return (
        ASLDataset(train_data, train_transform),
        ASLDataset(test_data, test_transform)
    )

# Training loop
def train_model():
    # Initialize
    train_set, test_set = prepare_data()
    train_loader = DataLoader(train_set, batch_size=CFG_ASLNet.BATCH_SIZE, 
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=CFG_ASLNet.BATCH_SIZE,
                            num_workers=4, pin_memory=True)
    
    model = ASLNet().to(__DEVICE__)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG_ASLNet.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    # Track metrics
    train_losses = []
    val_accuracies = []
    best_acc = 0.0
    
    # Create output directory
    plot_dir = Path("./training_plots")
    plot_dir.mkdir(exist_ok=True)
    
    # Training loop
    for epoch in range(CFG_ASLNet.EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG_ASLNet.EPOCHS}")
        
        for images, labels in progress_bar:
            images, labels = images.to(__DEVICE__), labels.to(__DEVICE__)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())
            
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_set)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(__DEVICE__), labels.to(__DEVICE__)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Store for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate validation accuracy
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # Plot sample predictions
        _plot_predictions(
            model=model,
            dataset=test_set,
            device=__DEVICE__,
            epoch=epoch+1,
            plot_dir=plot_dir,
            num_samples=6
        )        

        # Update scheduler
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(CFG_ASLNet.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), CFG_ASLNet.BEST_MODEL)
        
        # Printepoch summary
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_set):.4f} | " 
                f"Val Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%")

        # Plot metrics after each epoch
        _plot_metrics(train_losses, val_accuracies, plot_dir)
    
        # Final plots
        _plot_confusion_matrix(all_preds, all_labels, plot_dir)
    
if __name__ == "__main__":
    train_model()
    print("Training completed. Best model saved to:", CFG_ASLNet.BEST_MODEL)
