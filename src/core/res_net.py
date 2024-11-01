from src.utils.imports import layers, models, Model  # keras
from src.utils.imports import KerasTensor
import keras

class ResNet:
    def __init__(self, input_shape: tuple, num_classes_units: int = None, num_blocks: list[int] = [2, 2, 2, 2], include_top: bool = True):
        """
        Inicializa a arquitetura ResNet.
        
        Parâmetros:
        - input_shape: Dimensão de entrada das imagens.
        - num_classes_units: Número de classes (para a camada de classificação, se incluída).
        - num_blocks: Número de blocos residuais em cada estágio.
        - include_top: Se True, inclui a camada de classificação no topo do modelo.
        """
        self.include_top = include_top
        self.num_classes_units = num_classes_units
        self.model = self.build_resnet(input_shape, num_blocks)
        self.model.summary()

    def residual_block(self, x: KerasTensor, filters: int, kernel_size: int = 3, stride: int = 1) -> KerasTensor:
        shortcut = x  # Conexão de atalho

        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.01))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    def build_resnet(self, input_shape: tuple, num_blocks: list[int]) -> Model:
        """
        Constrói a arquitetura ResNet personalizada para classificação ou extração de características.
        
        Parâmetros:
        - input_shape: Dimensão de entrada das imagens.
        - num_blocks: Número de blocos residuais em cada estágio.
        
        Retorna:
        - Model: o modelo Keras da ResNet.
        """
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        #x = layers.Dropout(0.3)(x)  # Dropout para evitar overfitting

        # Stack residual blocks
        filters: int = 64
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                stride = 1 if j != 0 else (2 if i != 0 else 1)
                x = self.residual_block(x, filters, stride=stride)
            filters *= 2

        x = layers.GlobalAveragePooling2D()(x)
        if self.include_top and self.num_classes_units is not None:
            x = layers.Dense(units=self.num_classes_units, activation='softmax', bias_regularizer=keras.regularizers.l2(0.001))(x)
            #x = layers.Dropout(0.3)(x)  # Dropout adicional na camada de saída

        model = models.Model(inputs, x)
        return model