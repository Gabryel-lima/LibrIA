import cv2
import pandas as pd
import numpy as np
import os
import json
import keras
from keras import layers, Model
from src.core.GestureResNet import GestureResNet
from keras._tf_keras.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow as tf

# class DataGenerator(Sequence):
#     def __init__(self, data, labels, batch_size):
#         self.data = data
#         self.labels = labels
#         self.batch_size = batch_size

#     def __len__(self):
#         return int(np.ceil(len(self.data[0]) / self.batch_size))

#     def __getitem__(self, index):
#         batch_x1 = self.data[0][index * self.batch_size:(index + 1) * self.batch_size]
#         batch_x2 = self.data[1][index * self.batch_size:(index + 1) * self.batch_size]
#         batch_x3 = self.data[2][index * self.batch_size:(index + 1) * self.batch_size]
#         batch_y = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
#         return [batch_x1, batch_x2, batch_x3], batch_y

#     @staticmethod
#     def get_tf_dataset(data, labels, batch_size=64):
#         def generator():
#             for i in range(len(data[0])):
#                 yield [data[0][i], data[1][i], data[2][i]], labels[i]

#         # Definindo a assinatura de saída para o tf.data.Dataset
#         output_signature = (
#             (
#                 tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32),  # Imagem
#                 tf.TensorSpec(shape=(42,), dtype=tf.float32),        # Características dos gestos
#                 tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32)   # Pixels
#             ),
#             tf.TensorSpec(shape=(10, 32), dtype=tf.float32)          # Rótulos em sequência
#         )

#         dataset = tf.data.Dataset.from_generator(
#             generator,
#             output_signature=output_signature
#         )

#         dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         return dataset


class MultiModalGestureNet:
    def __init__(self, image_shape=(28, 28, 1), gesture_features_dim=42, num_blocks=[2, 2, 2, 2], num_classes=None):
        """
        Inicializa a classe para prever gestos com entradas de imagens e características dos gestos.
        
        Parâmetros:
        - image_shape (tuple): Dimensão de entrada das imagens, geralmente (altura, largura, canais).
        - gesture_features_dim (int): Número de valores na entrada das características dos gestos (ex: 42 para 21 pontos `(x, y)`).
        - num_blocks (list): Número de blocos residuais na ResNet.
        - num_classes (int): Número de classes de saída. Pode ser definido automaticamente durante o carregamento dos dados.
        """
        self.image_shape = image_shape
        self.gesture_features_dim = gesture_features_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.model = None

    # def get_data_generator(self, images, gesture_features, pixels, labels, batch_size=64):
    #     """
    #     Cria um DataGenerator para fornecer os dados em pequenos lotes durante o treinamento.
    #     """
    #     data = [images, gesture_features, pixels]
    #     return DataGenerator.get_tf_dataset(data, labels, batch_size)

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
        # Carregar dados de treino e teste a partir de arquivos CSV
        signals_train = pd.read_csv("E:\\libria\\data\\signals_train.csv")
        signals_test = pd.read_csv("E:\\libria\\data\\signals_test.csv")
        landmarks_train = pd.read_csv("E:\\libria\\data\\landmarks_train.csv")
        landmarks_test = pd.read_csv("E:\\libria\\data\\landmarks_test.csv")
        hands_train = pd.read_csv("E:\\libria\\data\\hands_train.csv")
        hands_test = pd.read_csv("E:\\libria\\data\\hands_test.csv")

        # Mesclar os dados de treino e teste em conjuntos únicos
        signals = pd.concat([signals_train, signals_test], ignore_index=True)
        landmarks = pd.concat([landmarks_train, landmarks_test], ignore_index=True)
        hands = pd.concat([hands_train, hands_test], ignore_index=True)

        # Extração dos rótulos e dados das características
        labels = signals['label'].values
        images = signals.drop('label', axis=1).values
        gesture_features = landmarks.drop('label', axis=1).iloc[:, :self.gesture_features_dim].values
        pixels = hands.drop('label', axis=1).values

        # Converter rótulos para valores numéricos adequados
        labels = pd.factorize(labels)[0]  # Converte rótulos categóricos para valores numéricos

        # Garantir que todos os conjuntos de dados tenham o mesmo número de amostras
        min_len = min(len(images), len(gesture_features), len(pixels), len(labels))

        images = images[:min_len]
        gesture_features = gesture_features[:min_len]
        pixels = pixels[:min_len]
        labels = labels[:min_len]

        # Balancear os dados antes de fazer a divisão
        labels, images, gesture_features, pixels = self.balance_classes(labels, images, gesture_features, pixels, min_count=5)

        # Redimensionar os dados de imagem para o formato (28, 28, 1)
        images = images.reshape((-1, 28, 28, 1))
        pixels = pixels.reshape((-1, 28, 28, 1))

        # Codificação one-hot para os rótulos
        self.num_classes = len(np.unique(labels))
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Ajuste para sequência temporal (10 passos)
        labels_seq = np.repeat(labels[:, np.newaxis, :], 10, axis=1)

        # Dividir dados em conjuntos de treino e teste
        (X_train_img, X_test_img,
         X_train_gesture_features, X_test_gesture_features,
         X_train_pixels, X_test_pixels,
         y_train_seq, y_test_seq) = train_test_split(
            images, gesture_features, pixels, labels_seq,
            test_size=0.2, random_state=42, stratify=labels
        )

        # Retornar dados em formato compacto
        return (X_train_img, X_train_gesture_features, X_train_pixels), (X_test_img, X_test_gesture_features, X_test_pixels), y_train_seq, y_test_seq

    def build_model(self):
        image_input = layers.Input(shape=self.image_shape)
        self.resnet = GestureResNet(self.image_shape, num_classes_units=self.num_classes, num_blocks=self.num_blocks, include_top=False)
        image_features = self.resnet.model(image_input)

        gesture_input = layers.Input(shape=(self.gesture_features_dim,))
        gesture_features = layers.BatchNormalization()(gesture_input)
        gesture_features = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(gesture_features)
        gesture_features = layers.BatchNormalization()(gesture_features)
        gesture_features = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(gesture_features)

        pixels_input = layers.Input(shape=self.image_shape)
        pixels_features = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pixels_input)
        pixels_features = layers.MaxPooling2D((2, 2))(pixels_features)
        pixels_features = layers.Flatten()(pixels_features)

        combined = layers.Concatenate()([image_features, gesture_features, pixels_features])
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(combined)
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)

        x = layers.RepeatVector(10)(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
        attention = layers.Attention()([x, x])

        x = layers.TimeDistributed(layers.Dense(32, activation='relu'))(attention)
        decoder = layers.GRU(128, return_sequences=True)(x)
        output = layers.TimeDistributed(layers.Dense(self.num_classes, activation='softmax'))(decoder)

        self.model = Model(inputs=[image_input, gesture_input, pixels_input], outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
