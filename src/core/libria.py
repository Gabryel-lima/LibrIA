from keras import layers, Model
from src.utils.imports import pd, np, to_categorical, train_test_split
from src.core.res_net import ResNet
from sklearn.utils import resample
import tensorflow as tf
import keras

class Libria:
    def __init__(self, image_shape=(28, 28, 1), landmark_dim=42, num_blocks=[2, 2, 2, 2], num_classes=None):
        """
        Inicializa a classe Libria para prever gestos com entradas de imagens e landmarks.
        
        Parâmetros:
        - image_shape: Dimensão de entrada das imagens.
        - landmark_dim: Número de valores na entrada de landmarks (ex: 42 para 21 pontos `(x, y)`).
        - num_blocks: Número de blocos residuais na ResNet.
        - num_classes: Número de classes de saída.
        """
        self.image_shape = image_shape
        self.landmark_dim = landmark_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.model = None
        self.resnet = None

    def load_data(self):
        """
        Carrega e processa as imagens e landmarks para treinamento.
        
        Retorna:
        - X_train_img, X_test_img, X_train_landmarks, X_test_landmarks, y_train_seq, y_test_seq: Dados divididos e ajustados.
        """
        # Carregar dados de imagem e landmarks
        image_data_train = pd.read_csv("E:\\libria\\data\\Signals\\sign_mnist_train\\sign_mnist_train.csv")
        image_data_test = pd.read_csv("E:\\libria\\data\\Signals\\sign_mnist_test\\sign_mnist_test.csv")
        landmark_data_train = pd.read_csv("E:\\libria\\data\\landmarks_hands_train.csv")
        landmark_data_test = pd.read_csv("E:\\libria\\data\\landmarks_hands_test.csv")

        # Mesclar e processar dados
        image_data = pd.concat([image_data_train, image_data_test], ignore_index=True)
        landmark_data = pd.concat([landmark_data_train, landmark_data_test], ignore_index=True)

        labels = image_data['label'].values
        images = image_data.drop('label', axis=1).values
        landmarks = landmark_data.drop('label', axis=1).iloc[:, :self.landmark_dim].values

        # Redimensionamento e normalização de imagens
        images = images.reshape(-1, *self.image_shape).astype('float32')
        images = (images - 0.5) * 2  # Normalizando para o intervalo [-1, 1]

        # Normalização dos landmarks para [0, 1] e adicionando ruído
        landmarks = landmarks.astype('float32') / 28  # Normalizando para a escala entre 0 e 1

        # Ajustar tamanhos desiguais de amostras entre imagens e landmarks
        if len(images) > len(landmarks):
            landmarks = resample(landmarks, replace=True, n_samples=len(images), random_state=42)
        elif len(landmarks) > len(images):
            images = resample(images, replace=True, n_samples=len(landmarks), random_state=42)
            labels = resample(labels, replace=True, n_samples=len(landmarks), random_state=42)

        # Codificação one-hot para os labels
        self.num_classes = np.max(labels) + 1 if self.num_classes is None else self.num_classes
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Ajuste para sequência temporal (10 passos)
        labels_seq = np.repeat(labels[:, np.newaxis, :], 10, axis=1)

        # Dividir dados em conjuntos de treino e teste
        X_train_img, X_test_img, X_train_landmarks, X_test_landmarks, y_train_seq, y_test_seq = train_test_split(
            images, landmarks, labels_seq, test_size=0.2, random_state=42, stratify=labels
        )

        return X_train_img, X_test_img, X_train_landmarks, X_test_landmarks, y_train_seq, y_test_seq

    def build_model(self):
        """
        Constrói o modelo de rede neural com entradas de imagem e landmarks, incluindo processamento sequencial.
        """
        # Entrada de imagem para extração de características com ResNet
        image_input = layers.Input(shape=self.image_shape)
        self.resnet = ResNet(self.image_shape, num_classes_units=self.num_classes, num_blocks=self.num_blocks, include_top=False)
        image_features = self.resnet.model(image_input)

        # Entrada de landmarks para processamento separado
        landmark_input = layers.Input(shape=(self.landmark_dim,))
        landmark_features = layers.BatchNormalization()(landmark_input)
        landmark_features = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(landmark_features)
        landmark_features = layers.BatchNormalization()(landmark_features)
        landmark_features = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(landmark_features)

        # Combinação das características de imagem e landmarks
        combined = layers.Concatenate()([image_features, landmark_features])
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(combined)
        #x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        #x = layers.Dropout(0.5)(x)

        # Sequenciamento com LSTM para captura de dependência temporal
        x = layers.RepeatVector(10)(x)  # Supõe que 10 é o tamanho da sequência
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

        # Attention Layer
        attention = layers.Attention()([x, x])

        # Dense layer after attention
        x = layers.Dense(32, activation='relu')(attention)
        #x = layers.Dropout(0.5)(x)  # Regularização adicional para melhorar a generalização

        # GRU layer for decoding
        decoder = layers.GRU(128, return_sequences=True)(x)
        #decoder = layers.Dropout(0.5)(decoder)

        output = layers.TimeDistributed(layers.Dense(self.num_classes, activation='softmax'))(decoder)

        # Corrigir a entrada do modelo
        self.model = Model(inputs=[image_input, landmark_input], outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
