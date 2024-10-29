from src.utils.imports import (
    pd, # import pandas as pd
    np, # import numpy as np
    to_categorical, # from sklearn.model_selection import train_test_split
    train_test_split, # from keras.src.utils.numerical_utils import to_categorical
)

from src.core.res_net import ResNet

class Libria:
    def __init__(self, input_shape: tuple[int, ...] = (), num_blocks: list[int] = [2, 2, 2, 2]):
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_classes = None
        self.ResNet = None

    def load_data(self) -> list:
        # Carregar o conjunto de dados de treinamento e teste
        train_data = pd.read_csv("E:\\libria\\data\\Signals\\sign_mnist_train\\sign_mnist_train.csv")
        test_data = pd.read_csv("E:\\libria\\data\\Signals\\sign_mnist_test\\sign_mnist_test.csv")

        # Combinar ambos para realizar uma divisão consistente
        data = pd.concat(objs=[train_data, test_data], ignore_index=True)

        # Separando as labels e as features
        labels = data['label'].values
        images = data.drop('label', axis=1).values

        # Redimensionando as imagens para 28x28 e normalizando os valores
        images = images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        # Definir o número de classes com base nas labels
        self.num_classes = np.max(labels) + 1
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Dividindo os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Inicializar o modelo após definir o número de classes
        self.ResNet = ResNet(self.input_shape, self.num_classes, self.num_blocks)

        return X_train, X_test, y_train, y_test
