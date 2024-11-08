import cv2
import pandas as pd
import numpy as np
import os
import keras
from keras import layers, Model
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow as tf
from src.utils.preprocessing import find_outliers_iqr_with_combined_histogram, verificar_normalizacao

from src.utils.imports import layers, models, Model  # keras
from src.utils.imports import KerasTensor
import keras

class GestureResNet:
    def __init__(self, image_input_shape: tuple, gesture_features_dim: int = 42, num_classes_units: int = None, num_blocks: list[int] = [2, 2, 2, 2]):
        """
        Inicializa a arquitetura GestureResNet.
        """
        self.num_classes_units = num_classes_units
        self.gesture_features_dim = gesture_features_dim
        self.model = self.build_model(image_input_shape, num_blocks)

    def residual_block(self, x, filters: int = 75, kernel_size: tuple[int, int] | int = 3, stride: int = 1):
        """
        Cria um bloco residual.
        """
        shortcut = x

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

    def build_model(self, input_shape: tuple, num_blocks: list[int]):
        """
        Constrói a arquitetura GestureResNet.
        """
        inputs = layers.Input(shape=input_shape)

        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        filters: int = 8
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                stride = 1 if j != 0 else (2 if i != 0 else 1)
                x = self.residual_block(x, filters, stride=stride)
            filters *= 2

        x = layers.GlobalAveragePooling2D()(x)

        model = models.Model(inputs, x)
        return model

class MultiModalGestureNet:
    def __init__(self, image_shape=(28, 28, 1), gesture_features_dim=42, num_blocks=[2, 2, 2, 2], num_classes=None):
        self.image_shape = image_shape
        self.gesture_features_dim = gesture_features_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.model = None

    @staticmethod
    def balance_classes(labels, images, gesture_features, pixels, min_count=2):
        df = pd.DataFrame({
            'labels': labels,
            'images': list(images),
            'gesture_features': list(gesture_features),
            'pixels': list(pixels)
        })

        classes = df['labels'].unique()
        balanced_data = []

        for label in classes:
            class_data = df[df['labels'] == label]
            if len(class_data) < min_count:
                class_data = resample(class_data, replace=True, n_samples=min_count, random_state=42)
            balanced_data.append(class_data)

        balanced_df = pd.concat(balanced_data, ignore_index=True)

        return (balanced_df['labels'].values,
                np.stack(balanced_df['images'].values),
                np.stack(balanced_df['gesture_features'].values),
                np.stack(balanced_df['pixels'].values))

    def load_data(self):
        signals_train = pd.read_csv("E:\\libria\\data\\signals_train.csv")
        signals_test = pd.read_csv("E:\\libria\\data\\signals_test.csv")
        landmarks_train = pd.read_csv("E:\\libria\\data\\landmarks_train.csv")
        landmarks_test = pd.read_csv("E:\\libria\\data\\landmarks_test.csv")
        hands_train = pd.read_csv("E:\\libria\\data\\hands_train.csv")
        hands_test = pd.read_csv("E:\\libria\\data\\hands_test.csv")

        signals = pd.concat([signals_train, signals_test], ignore_index=True)
        landmarks = pd.concat([landmarks_train, landmarks_test], ignore_index=True)
        hands = pd.concat([hands_train, hands_test], ignore_index=True)

        labels = signals['label'].values
        images = signals.drop('label', axis=1).values.astype('float32')
        gesture_features = landmarks.drop('label', axis=1).iloc[:, :self.gesture_features_dim].values.astype('float32')
        pixels = hands.drop('label', axis=1).values.astype('float32')

        # Normalização dos dados
        images = (images - images.mean(axis=0)) / (images.std(axis=0) + 1e-8)
        pixels = (pixels - pixels.mean(axis=0)) / (pixels.std(axis=0) + 1e-8)
        gesture_features = (gesture_features - gesture_features.mean(axis=0)) / (gesture_features.std(axis=0) + 1e-8)

        # Converter rótulos para valores numéricos adequados
        labels = pd.factorize(labels)[0]

        # Garantir que todos os conjuntos de dados tenham o mesmo número de amostras
        min_len = min(len(images), len(gesture_features), len(pixels), len(labels))
        images = images[:min_len]
        gesture_features = gesture_features[:min_len]
        pixels = pixels[:min_len]
        labels = labels[:min_len]

        # Balancear os dados antes de fazer a divisão
        labels, images, gesture_features, pixels = self.balance_classes(labels, images, gesture_features, pixels, min_count=2)

        # Redimensionar os dados de imagem para o formato (28, 28, 1)
        images = images.reshape((-1, 28, 28, 1))
        pixels = pixels.reshape((-1, 28, 28, 1))

        # Codificação one-hot para os rótulos
        self.num_classes = len(np.unique(labels))
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Ajuste para sequência temporal (10 passos)
        labels_seq = np.repeat(labels[:, np.newaxis, :], 32, axis=1)

        # Dividir dados em conjuntos de treino e teste
        (X_train_img, X_test_img,
         X_train_gesture_features, X_test_gesture_features,
         X_train_pixels, X_test_pixels,
         y_train, y_test) = train_test_split(
            images, gesture_features, pixels, labels_seq,
            test_size=0.2, random_state=42, stratify=labels
        )

        return (X_train_img, X_train_gesture_features, X_train_pixels), (X_test_img, X_test_gesture_features, X_test_pixels), y_train, y_test

    def build_model(self):
        # Entrada de imagem para extração de características com ResNet
        image_input = layers.Input(shape=self.image_shape)
        self.resnet = GestureResNet(self.image_shape, num_classes_units=self.num_classes, num_blocks=self.num_blocks)
        image_features = self.resnet.model(image_input)
        image_features = layers.BatchNormalization()(image_features)

        # Entrada das características dos gestos para processamento separado
        gesture_input = layers.Input(shape=(self.gesture_features_dim, ))
        gesture_features = layers.BatchNormalization()(gesture_input)
        gesture_features = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(gesture_features)
        gesture_features = layers.Dropout(0.3)(gesture_features)
        gesture_features = layers.BatchNormalization()(gesture_features)
        gesture_features = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(gesture_features)

        # Entrada de pixels para processamento separado
        pixels_input = layers.Input(shape=self.image_shape)
        pixels_features = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pixels_input)
        pixels_features = layers.Dropout(0.3)(pixels_features)
        pixels_features = layers.MaxPooling2D((2, 2))(pixels_features)
        pixels_features = layers.GlobalAveragePooling2D()(pixels_features)
        pixels_features = layers.BatchNormalization()(pixels_features)

        # Combinação das características de imagem, gestos e pixels
        combined = layers.Concatenate()([image_features, gesture_features, pixels_features])
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(combined)
        x = layers.Dropout(0.4)(x)

        # Resto da rede
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))(x)
        x = layers.RepeatVector(32)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
        attention = layers.Attention()([x, x])
        attention = layers.Dropout(0.3)(attention)
        x = layers.TimeDistributed(layers.Dense(32, activation='relu'))(attention)
        decoder = layers.GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
        output = layers.TimeDistributed(layers.Dense(self.num_classes, activation='softmax'))(decoder)

        # Definindo o modelo com todas as entradas
        self.model = Model(inputs=[image_input, gesture_input, pixels_input], outputs=output)

        # Compilando o modelo
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy'
            ],
        )

