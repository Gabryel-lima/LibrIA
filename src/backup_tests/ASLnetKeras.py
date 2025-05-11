"""A ideia é a gente unir a arquitetura ao notebook do Kaggle, que usa o VGG-16. 
A gente vai passar essa arquitetura do torch para receber RGB em vez da escala de cinza. 
E vamos aumentar a capacidade da rede, porque a gente vai rodar ela pelo próprio servidor do Kaggle."""

# Implementação em Keras (tf.keras.Model)
# Está arquitetura será levada para o a gpu P-100 da Google Colab

import tensorflow as tf
from keras.api import layers, Model    # from tensorflow.keras import layers, Model

class ASLNetKeras(Model):
    def __init__(self, img_size, num_classes, dropout_rate):
        super(ASLNetKeras, self).__init__()
        # Bloco convolucional 1
        self.conv1 = layers.Conv2D(32, 3, padding='same', input_shape=(img_size, img_size, 1))
        self.bn1   = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.pool1 = layers.MaxPooling2D(2)
        # Bloco convolucional 2
        self.conv2 = layers.Conv2D(64, 3, padding='same')
        self.bn2   = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.pool2 = layers.MaxPooling2D(2)
        # Bloco convolucional 3
        self.conv3 = layers.Conv2D(128, 3, padding='same')
        self.bn3   = layers.BatchNormalization()
        self.relu3 = layers.ReLU()
        self.pool3 = layers.MaxPooling2D(2)
        # Classificador
        self.flatten    = layers.Flatten()
        self.fc1        = layers.Dense(512)
        self.relu_fc1   = layers.ReLU()
        self.dropout    = layers.Dropout(dropout_rate)
        self.out_logits = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x, training=training)
        return self.out_logits(x)

# Exemplo de instância:
model_keras = ASLNetKeras(
    img_size=224,
    num_classes=29,
    dropout_rate=0.5
)

