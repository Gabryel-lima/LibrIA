"""A ideia é a gente unir a arquitetura ao notebook do Kaggle, que usa o VGG-16. 
A gente vai passar essa arquitetura do torch para receber RGB em vez da escala de cinza. 
E vamos aumentar a capacidade da rede, porque a gente vai rodar ela pelo próprio servidor do Kaggle."""

# Implementação em Keras (tf.keras.Model)
# Está arquitetura será levada para o a gpu P-100 da Google Colab

import tensorflow as tf
from keras.api import layers, Model   # from tensorflow.keras import layers, Model --> importação alternativa
from keras.api.applications import VGG16    # from tensorflow.keras.applications import VGG16 --> importação alternativa

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

# Exemplo de instância:
model_keras = ASLNetKeras(
    img_size=224,
    num_classes=29,
    dropout_rate=0.5
)
