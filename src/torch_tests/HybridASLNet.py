import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import cv2

from src.torch_tests.conf import CFG_HybridASLNet, __DEVICE__

# --- Dataset ---
class ASLDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')  # <- Corrigido para RGB
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return self[(idx + 1) % len(self)]

# --- Modelo Híbrido com VGG16 congelada ---
class HybridASLNet(nn.Module):
    def __init__(self, num_classes=29, dropout=0.5):
        super(HybridASLNet, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in vgg.features.parameters():
            param.requires_grad = False

        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- GradCAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

# --- Preparo dos dados ---
def prepare_data():
    data = []
    for label_idx, label in enumerate(CFG_HybridASLNet.LABELS):
        class_dir = os.path.join(CFG_HybridASLNet.DATA_DIR, label)
        if not os.path.exists(class_dir):
            continue
        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                data.append((os.path.join(class_dir, file), label_idx))

    train_data, test_data = train_test_split(
        data, test_size=0.2, stratify=[d[1] for d in data], random_state=42
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.Resize((CFG_HybridASLNet.IMG_SIZE, CFG_HybridASLNet.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((CFG_HybridASLNet.IMG_SIZE, CFG_HybridASLNet.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return ASLDataset(train_data, train_transform), ASLDataset(test_data, test_transform)

# --- Treinamento ---
def train_model():
    train_set, test_set = prepare_data()
    train_loader = DataLoader(train_set, batch_size=CFG_HybridASLNet.BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=CFG_HybridASLNet.BATCH_SIZE, num_workers=1, pin_memory=True)

    model = HybridASLNet(num_classes=CFG_HybridASLNet.NUM_CLASSES, dropout=CFG_HybridASLNet.DROPOUT).to(__DEVICE__)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG_HybridASLNet.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    train_losses, val_accuracies = [], []
    best_acc = 0.0
    plot_dir = Path("./training_plots"); plot_dir.mkdir(exist_ok=True)

    for epoch in range(CFG_HybridASLNet.EPOCHS):
        model.train(); running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG_HybridASLNet.EPOCHS}")
        for images, labels in progress_bar:
            images, labels = images.to(__DEVICE__), labels.to(__DEVICE__)
            optimizer.zero_grad(); outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_set)
        train_losses.append(epoch_loss)

        model.eval(); correct = total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(__DEVICE__), labels.to(__DEVICE__)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0); correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(CFG_HybridASLNet.BEST_MODEL), exist_ok=True)
            torch.save(model.state_dict(), CFG_HybridASLNet.BEST_MODEL)

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()
    print("Treinamento concluído. Melhor modelo salvo em:", CFG_HybridASLNet.BEST_MODEL)
