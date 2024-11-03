from src.utils.imports import layers, models, Model  # keras
from src.utils.imports import KerasTensor
import keras

class GestureResNet:
    def __init__(self, image_input_shape: tuple, gesture_features_dim: int = 42, num_classes_units: int = None, num_blocks: list[int] = [2, 2, 2, 2], include_top: bool = True):
        """
        Inicializa a arquitetura GestureResNet.
        
        Parâmetros:
        - image_input_shape (tuple): Dimensão de entrada das imagens.
        - gesture_features_dim (int): Dimensão das características dos gestos (ex: 42 para 21 pontos `(x, y)`).
        - num_classes_units (int): Número de classes (para a camada de classificação, se incluída).
        - num_blocks (list[int]): Número de blocos residuais em cada estágio.
        - include_top (bool): Se True, inclui a camada de classificação no topo do modelo.
        """
        self.include_top = include_top
        self.num_classes_units = num_classes_units
        self.gesture_features_dim = gesture_features_dim
        self.model = self.build_model(image_input_shape, num_blocks)
        self.model.summary()

    def residual_block(self, x: KerasTensor, filters: int = 75, kernel_size: tuple[int, int] | int = 3, stride: int = 1) -> KerasTensor:
        """
        Cria um bloco residual.
        
        Parâmetros:
        - x (KerasTensor): Entrada para o bloco residual.
        - filters (int): Número de filtros para as camadas convolucionais.
        - kernel_size (tuple[int, int] | int): Tamanho do kernel para a convolução.
        - stride (int): Stride para a primeira camada convolucional do bloco.

        Retorna:
        - KerasTensor: Saída do bloco residual após a conexão de atalho e operações ReLU.
        """
        # Conexão de atalho
        shortcut = x

        # Primeira camada convolucional do bloco
        x = layers.Conv2D(
            filters, 
            kernel_size=kernel_size, 
            strides=stride, 
            padding='same', 
            use_bias=False, 
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Segunda camada convolucional do bloco
        x = layers.Conv2D(
            filters, 
            kernel_size=kernel_size, 
            strides=1, 
            padding='same', 
            use_bias=False, 
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(x)
        x = layers.BatchNormalization()(x)

        # Ajuste do atalho se necessário
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, 
                kernel_size=1, 
                strides=stride, 
                padding='same', 
                use_bias=False, 
                kernel_regularizer=keras.regularizers.l2(0.01)
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Adiciona a conexão de atalho e aplica ReLU
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    def build_model(self, input_shape: tuple, num_blocks: list[int]) -> Model:
        """
        Constrói a arquitetura GestureResNet personalizada para classificação ou extração de características.
        
        Parâmetros:
        - input_shape (tuple): Dimensão de entrada das imagens.
        - num_blocks (list[int]): Número de blocos residuais em cada estágio.

        Retorna:
        - Model: O modelo Keras daGestureResNet.
        """
        # Entrada do modelo
        inputs = layers.Input(shape=input_shape)

        # Primeira camada convolucional
        x = layers.Conv2D(
            64, 
            kernel_size=3, 
            strides=1, 
            padding='same', 
            use_bias=False, 
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Empilhar blocos residuais
        filters = 64
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                stride = 1 if j != 0 else (2 if i != 0 else 1)
                x = self.residual_block(x, filters, stride=stride)
            filters *= 2

        # Camada de pooling global
        x = layers.GlobalAveragePooling2D()(x)

        # Adiciona a camada de classificação se include_top for True
        if self.include_top and self.num_classes_units is not None:
            x = layers.Dense(
                units=self.num_classes_units, 
                activation='softmax', 
                bias_regularizer=keras.regularizers.l2(0.001)
            )(x)

        # Construção do modelo
        model = models.Model(inputs, x)
        return model
