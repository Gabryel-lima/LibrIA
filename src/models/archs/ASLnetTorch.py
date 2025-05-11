"""A ideia é a gente unir a arquitetura ao notebook do Kaggle, que usa o VGG-16. 
A gente vai passar essa arquitetura do torch para receber RGB em vez da escala de cinza. 
E vamos aumentar a capacidade da rede, porque a gente vai rodar ela pelo próprio servidor do Kaggle."""

# Implementação em torch (torch.nn.Module)
# Está arquitetura será levada para o a gpu P-100 da Google Colab

import torch
import torch.nn as nn
from torchvision import models

# Model base class
class ASLNetVGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1) Carrega VGG-16 pré-treinado (aceita 3 canais RGB)
        vgg = models.vgg16(pretrained=True)
        
        # 2) Opcional: descongelar todos ou apenas as camadas mais profundas
        for param in vgg.features.parameters():
            param.requires_grad = False  # congela convolucionais iniciais
        # Exemplo: liberar o bloco 5
        for param in vgg.features[24:].parameters():
            param.requires_grad = True

        # 3) Aumentar a capacidade do classificador
        #    – original: [4096 → 4096 → num_classes]
        #    – modificado: [4096 → 2048 → 1024 → num_classes]
        self.features   = vgg.features
        self.avgpool    = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 2048),       # camada extra
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),       # camada extra
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, num_classes) # saída final
        )

    def forward(self, x):
        x = self.features(x)             # convoluções VGG
        x = self.avgpool(x)              # pooling adaptado
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Exemplo de instância:
model_keras = ASLNetVGG(
    num_classes=29
)
