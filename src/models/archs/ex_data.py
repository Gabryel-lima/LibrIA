"""Exeplos de architeturas de CNNs para o dataset ASL"""

# imports global
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

# ==================================================== # 

# Implementação em torch (torch.nn.Module)
# Está arquitetura será levada para o a gpu P-100 da Google Colab

import torch
import torch.nn as nn
from torchvision import models

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
            image = Image.open(img_path).convert('RGB') # Convert to RGB
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return self[(idx + 1) % len(self)]  # Skip corrupted files

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
    
# continua ... transformação de dados

# Exemplo de instância:
model_keras = ASLNetVGG(
    num_classes=29
)

# ==================================================== #

# Implementação em Keras (tf.keras.Model)
# Está arquitetura será levada para o a gpu P-100 da Google Colab

import tensorflow as tf
from keras.api import layers, Model    # from tensorflow.keras import layers, Model --> importação alternativa
from keras.api.applications import VGG16    # from tensorflow.keras.applications import VGG16 --> importação alternativa

# TODO Adaptando dataset pelo torch

class ASLNetKeras(Model):
    def __init__(self, img_size, num_classes, dropout_rate=0.5):
        super().__init__()
        # 1) Backbone VGG16 sem topo
        base = VGG16(weights='imagenet',
                     include_top=False,
                     input_shape=(img_size, img_size, 3))
        # 2) Congela layers iniciais, libera bloco 5 (camadas 15–18 do base)
        for layer in base.layers[:-4]:
            layer.trainable = False
        for layer in base.layers[-4:]:
            layer.trainable = True
        self.backbone = base
        self.flatten  = layers.Flatten()
        # 3) Classificador equivalente
        self.fc1 = layers.Dense(4096, activation='relu')
        self.do1 = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(2048, activation='relu')
        self.do2 = layers.Dropout(dropout_rate)
        self.fc3 = layers.Dense(1024, activation='relu')
        self.do3 = layers.Dropout(dropout_rate)
        self.out = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        x = self.flatten(x)
        x = self.fc1(x); x = self.do1(x, training=training)
        x = self.fc2(x); x = self.do2(x, training=training)
        x = self.fc3(x); x = self.do3(x, training=training)
        return self.out(x)

# continua ... transformação de dados

# instancia:
model_keras = ASLNetKeras(img_size=224, num_classes=29, dropout_rate=0.5)

# ===================================================== #

# Implementação em TensorFlow “baixo nível” (tf.Module + tf.nn)
# Está arquitetura será levada para o a gpu P-100 da Google Colab

import tensorflow as tf

# TODO Adaptando dataset pelo torch

class ASLNetTF(tf.Module):
    def __init__(self, img_size, num_classes, dropout_rate=0.5, name=None):
        super().__init__(name=name)
        init = tf.initializers.GlorotUniform()
        zeros = tf.zeros_initializer()

        # Variáveis do backbone VGG16 (poderiam ser carregadas via TF Hub,
        # mas aqui consideramos apenas o classificador)

        flat_dim = (img_size//32)*(img_size//32)*512  # 7×7×512 = 25088

        # Classificador: 25088→4096→2048→1024→num_classes
        self.w1 = tf.Variable(init([flat_dim, 4096]), name='w1')
        self.b1 = tf.Variable(zeros([4096]),      name='b1')
        self.w2 = tf.Variable(init([4096,    2048]), name='w2')
        self.b2 = tf.Variable(zeros([2048]),      name='b2')
        self.w3 = tf.Variable(init([2048,    1024]), name='w3')
        self.b3 = tf.Variable(zeros([1024]),      name='b3')
        self.w4 = tf.Variable(init([1024, num_classes]), name='w4')
        self.b4 = tf.Variable(zeros([num_classes]),      name='b4')

        self.dropout_rate = dropout_rate

    @tf.function
    def __call__(self, x, training=False):
        # x deve vir de VGG16 sem topo: shape=(B,7,7,512)
        x = tf.reshape(x, [tf.shape(x)[0], -1])  # (B,25088)
        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        if training: x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = tf.matmul(x, self.w2) + self.b2
        x = tf.nn.relu(x)
        if training: x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = tf.matmul(x, self.w3) + self.b3
        x = tf.nn.relu(x)
        if training: x = tf.nn.dropout(x, rate=self.dropout_rate)

        logits = tf.matmul(x, self.w4) + self.b4
        return logits
    
# continua ... transformação de dados

# Instância (após extrair features via VGG16):
model_tf = ASLNetTF(
    img_size=224, 
    num_classes=29, 
    dropout_rate=0.5
)
