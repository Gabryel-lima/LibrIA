from src.utils.imports import pd, np, to_categorical, train_test_split
from src.core.res_net import ResNet
from keras import layers, Model
from sklearn.utils import resample

class Libria:
    def __init__(self, image_shape=(28, 28, 1), landmark_dim=42, num_blocks=[2, 2, 2, 2], num_classes=None):
        """
        Inicializa a classe Libria para prever gestos com entradas de imagens e landmarks.
        
        Parâmetros:
        - image_shape: Dimensão da imagem de entrada.
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
        - X_train_img, X_test_img, X_train_landmarks, X_test_landmarks, y_train, y_test: Dados divididos.
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

        # Ajustar tamanhos desiguais de amostras entre imagens e landmarks
        if len(images) > len(landmarks):
            landmarks = resample(landmarks, replace=True, n_samples=len(images), random_state=42)
        elif len(landmarks) > len(images):
            images = resample(images, replace=True, n_samples=len(landmarks), random_state=42)
            labels = resample(labels, replace=True, n_samples=len(landmarks), random_state=42)

        # Redimensionamento e normalização de imagens
        images = images.reshape(-1, *self.image_shape).astype('float32') / 255.0

        # Codificação one-hot para os labels
        self.num_classes = np.max(labels) + 1 if self.num_classes is None else self.num_classes
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Dividir dados em conjuntos de treino e teste
        return train_test_split(
            images, landmarks, labels, test_size=0.2, random_state=42, stratify=labels
        )

    def build_model(self):
        """
        Constrói o modelo de rede neural com entradas de imagem e landmarks.
        """
        # Entrada de imagem para extração de características com ResNet
        image_input = layers.Input(shape=self.image_shape)
        self.resnet = ResNet(self.image_shape, num_classes_units=self.num_classes, num_blocks=self.num_blocks, include_top=False)
        image_features = self.resnet.model(image_input)

        # Entrada de landmarks para processamento separado
        landmark_input = layers.Input(shape=(self.landmark_dim,))
        landmark_features = layers.Dense(64, activation='relu')(landmark_input)
        landmark_features = layers.Dense(32, activation='relu')(landmark_features)

        # Combinação das características de imagem e landmarks
        combined = layers.Concatenate()([image_features, landmark_features])
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dense(64, activation='relu')(x)

        # Camada de saída para classificação
        output = layers.Dense(self.num_classes, activation='softmax')(x)

        # Modelo final
        self.model = Model(inputs=[image_input, landmark_input], outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
