from src.core.res_net import Res_Net
from sklearn.model_selection import train_test_split
from keras.src.utils.numerical_utils import to_categorical
import pandas as pd
import numpy as np


class Libria:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.num_classes = None
        self.res_net = None

    def load_data(self):
        # Carregar o conjunto de dados de treinamento e teste
        train_data = pd.read_csv("E:\\libria\\data\\Signals\\sign_mnist_train\\sign_mnist_train.csv")
        test_data = pd.read_csv("E:\\libria\\data\\Signals\\sign_mnist_test\\sign_mnist_test.csv")

        # Combinar ambos para realizar uma divisão consistente
        data = pd.concat([train_data, test_data], ignore_index=True)

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
        self.res_net = Res_Net(self.input_shape, self.num_classes)

        return X_train, X_test, y_train, y_test
    
