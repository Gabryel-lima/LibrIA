from src.utils.imports import pd, np, to_categorical, train_test_split
from src.core.res_net import ResNet
from keras import layers, Model, Input

class Libria:
    def __init__(self, image_shape=(28, 28, 1), landmark_dim=42, num_blocks=[2, 2, 2, 2]):
        """
        Inicializa a classe Libria para prever gestos com entradas de imagens e landmarks.
        
        Parâmetros:
        - image_shape: Dimensão da imagem de entrada.
        - landmark_dim: Número de valores na entrada de landmarks (ex: 42 para 21 pontos `(x, y)`).
        - num_blocks: Número de blocos residuais na ResNet.
        """
        self.image_shape = image_shape
        self.landmark_dim = landmark_dim
        self.num_blocks = num_blocks
        self.num_classes = None
        self.model = None

    def load_data(self):
        """
        Carrega imagens e landmarks para o treinamento.
        
        Retorna:
        - X_train_img, X_test_img, X_train_landmarks, X_test_landmarks, y_train, y_test: Conjuntos de dados divididos.
        """
        # Carregar dados de imagem e landmarks
        image_data_train = pd.read_csv("E:\\libria\\data\\Signals\\sign_mnist_train\\sign_mnist_train.csv")
        image_data_test = pd.read_csv("E:\\libria\\data\\Signals\\sign_mnist_test\\sign_mnist_test.csv")
        
        landmark_data_train = pd.read_csv("E:\\libria\\data\\landmarks_hands_train.csv")
        landmark_data_test = pd.read_csv("E:\\libria\\data\\landmarks_hands_test.csv")

        # Mesclar conjuntos de treinamento e teste
        image_data = pd.concat([image_data_train, image_data_test], ignore_index=True)
        landmark_data = pd.concat([landmark_data_train, landmark_data_test], ignore_index=True)

        # Separar labels, imagens e landmarks
        labels = image_data['label'].values
        images = image_data.drop('label', axis=1).values
        landmarks = landmark_data.drop('label', axis=1).values  # Supondo que landmarks também tenham uma coluna 'label'

        # Redimensionar e normalizar imagens
        images = images.reshape(-1, *self.image_shape).astype('float32') / 255.0

        # Convertendo labels para one-hot encoding
        self.num_classes = np.max(labels) + 1
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Dividir em conjuntos de treino e teste para imagens e landmarks
        X_train_img, X_test_img, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        X_train_landmarks, X_test_landmarks = train_test_split(landmarks, test_size=0.2, random_state=42)

        return X_train_img, X_test_img, X_train_landmarks, X_test_landmarks, y_train, y_test

    def build_model(self):
        """
        Constrói o modelo com entradas de imagens e landmarks.
        """
        # Caminho de imagem com ResNet, sem camada de classificação
        image_input = layers.Input(shape=self.image_shape)
        resnet = ResNet(self.image_shape, num_classes_units=self.num_classes, num_blocks=self.num_blocks, include_top=False)
        image_features = resnet.model(image_input)

        # Caminho de landmarks com uma rede totalmente conectada
        landmark_input = layers.Input(shape=(self.landmark_dim,))
        landmark_features = layers.Dense(64, activation='relu')(landmark_input)
        landmark_features = layers.Dense(32, activation='relu')(landmark_features)

        # Combinação das duas saídas
        combined = layers.Concatenate()([image_features, landmark_features])
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dense(64, activation='relu')(x)

        # Camada de saída para classificação final
        output = layers.Dense(self.num_classes, activation='softmax')(x)

        # Modelo final com duas entradas
        self.model = Model(inputs=[image_input, landmark_input], outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
