import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, img_labels, transform=None, target_transform=None):
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        image_path, label = self.img_labels[idx]
        image = Image.open(image_path).convert('L')  # Converte para escala de cinza (1 canal)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label = torch.tensor(label)  # Cria um tensor explícito para o rótulo

        return image, label

def datasset(save_path='E:\\Projects\\libria\\transformed_data.pth'):
    if os.path.exists(save_path):
        print(f'Loading: {save_path}...')
        return torch.load(save_path, weights_only=False)

    print('Transforming data...')
    
    IMG_DIR = 'E:\\Projects\\libria\\data\\asl_hands\\ASL_Alphabet_Dataset'

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])

    all_data = []
    label_to_idx = {}
    idx = 0
    
    # Walk through the directories and collect all images and their labels
    for root, dirs, files in os.walk(os.path.join(IMG_DIR, 'asl_alphabet_train')):
        if dirs:  # Verifica se o diretório tem subdiretórios (pastas de classe)
            for dir_name in dirs:
                label = dir_name
                if label not in label_to_idx:
                    label_to_idx[label] = idx
                    idx += 1
                label_idx = label_to_idx[label]
                class_dir = os.path.join(root, dir_name)
                print(f"Processando diretório: {class_dir}")  # Verificação
                for file in os.listdir(class_dir):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        image_path = os.path.join(class_dir, file)
                        all_data.append((image_path, label_idx))

    # Check how many images we have
    print(f'Total de imagens encontradas: {len(all_data)}')

    # Split data into train and test
    split_index = int(len(all_data) * 0.8)  # 80% for training, 20% for testing
    training_data_raw = all_data[:split_index]
    test_data_raw = all_data[split_index:]
    
    # Create the datasets
    training_data = CustomImageDataset(training_data_raw, transform=transform)
    test_data = CustomImageDataset(test_data_raw, transform=transform)
    
    print(f'Número de amostras de treino: {len(training_data)}')
    print(f'Número de amostras de teste: {len(test_data)}')
    
    # Save data
    print(f'Saving: {save_path}...')
    torch.save((training_data, test_data), save_path)

    return training_data, test_data

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
    
class Net(nn.Module):
    def __init__(self, num_classe: int = 29):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # Aplica uma convolução 2D
        self.pool = nn.MaxPool2d(2, 2) # Aplica uma média no grupo local em 2D
        self.conv2 = nn.Conv2d(6, 16, 5) # Dobrando as saídas
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Aplica uma camada totalmente conectada para combinar as características extraídas em representações abstratas
        self.fc2 = nn.Linear(120, 84) # Reduz a dimensionalidade para comprimir e refinar as representações
        self.fc3 = nn.Linear(84, num_classe) # Conecta a saída ao número de classes para a tarefa de classificação

    def forward(self, x):
        # Propagação 
        x = self.pool(F.relu(self.conv1(x)))  # Convolução + ReLU + MaxPooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolução + ReLU + MaxPooling
        x = x.view(-1, 16 * 5 *5) # Achatamento de 3d para 1d
        x = F.relu(self.fc1(x)) # Primeira camada totalmente conectada com ReLU
        x = F.relu(self.fc2(x)) # Segunda camada totalmente conectada com ReLU
        x = self.fc3(x) # Camada de saída
        return x

if __name__ == '__main__':
    # load data
    training_data, test_data = datasset()
    
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=2)
    
    print(f'Número de amostras de treino: {len(training_data)}')
    print(f'Número de amostras de teste: {len(test_data)}')
    
    # Exibe imagens do conjunto de treino
    # imshow(training_data)
    
    # Mostra batch do DataLoader
    train_features, train_labels = next(iter(train_dataloader))
    
    # after normalize
    # print(f"Labels batch shape: {len(train_labels)}")
    # print(f"First label: {train_labels[0]}")  # Mostra o primeiro rótulo
    
    # img = train_features[0].squeeze(0)
    # label = train_labels[0]
    # plt.imshow(img, cmap='gray')
    # plt.title(f'Label: {label}')
    # plt.show()
    
    # variation of LeNet
    net = Net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(12):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 2000}')
                running_loss = 0.0

    # Monitor accuracy for each epoch to track progress
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    torch.save(net, 'E:\\Projects\\libria\\model_full.pth')
    torch.save(net.state_dict(), 'E:\\Projects\\libria\\model_weights.pth')
    print(f'Accuracy on training data after epoch {epoch + 1}: {100 * correct / total}%')
    
    # Exibir uma imagem de predição e seu rótulo
    sample_idx = torch.randint(len(train_dataloader.dataset), size=(1,)).item()
    image, label = train_dataloader.dataset[sample_idx]
    image = image.squeeze(0)  # Remover o canal extra para visualização

    # Inferência com a imagem selecionada
    output = net(image.unsqueeze(0))  # Adicionar um lote fictício
    _, predicted = torch.max(output.data, 1)

    # Exibir imagem e classe prevista
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted: {predicted.item()}, Actual: {label.item()}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
