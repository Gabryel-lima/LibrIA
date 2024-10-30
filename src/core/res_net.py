from src.utils.imports import (
    layers, models, Model  # keras
)
from src.utils.imports import KerasTensor

class ResNet:
    def __init__(self, input_shape: tuple, num_classes_units: int, num_blocks: list[int] = [2, 2, 2, 2]):
        # Inicializar o modelo com a arquitetura ResNet
        self.model = self.build_resnet(input_shape, num_classes_units, num_blocks, activation_dense="softmax")

    # Função para criar um bloco residual
    def residual_block(self, x: KerasTensor, filters: int, kernel_size: int = 3, stride: int = 1) -> KerasTensor:
        shortcut = x  # Conexão de atalho

        # Primeira camada convolucional do bloco
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Segunda camada convolucional do bloco
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # Ajuste do atalho se a dimensão não corresponder
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Adicionando a conexão de atalho e ativação final
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    # Função para construir a arquitetura ResNet personalizada com Keras
    def build_resnet(self, input_shape: tuple, num_classes_units: int, num_blocks: list[int], activation_dense: str = 'softmax') -> Model:
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Empilhando os blocos residuais
        filters: int = 64
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                stride: int = 1 if j != 0 else (2 if i != 0 else 1)
                x = self.residual_block(x, filters, stride=stride)
            filters *= 2

        # Camadas finais de pooling e classificação
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(units=num_classes_units, activation=activation_dense)(x)
        model = models.Model(inputs, outputs)

        return model
