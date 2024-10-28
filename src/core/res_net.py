import numpy as np
from keras import layers, models


class Res_Net:
    def __init__(self, input_shape, num_classes):
        self.model = self.ResNet(input_shape, num_classes)

    # Função para o bloco residual
    def residual_block(self, inputs, filters, stride=1):
        shortcut = inputs
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (3, 3), strides=1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or inputs.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False)(inputs)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    # Função para criar a ResNet
    def ResNet(self, input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, (3, 3), strides=1, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = self.residual_block(x, 64, stride=1)
        x = self.residual_block(x, 64, stride=1)
        x = self.residual_block(x, 128, stride=2)
        x = self.residual_block(x, 128, stride=1)
        x = self.residual_block(x, 256, stride=2)
        x = self.residual_block(x, 256, stride=1)
        x = self.residual_block(x, 512, stride=2)
        x = self.residual_block(x, 512, stride=1)
        
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model