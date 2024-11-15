import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import keras
from keras import layers, Model
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [batch_size[..., tf.newaxis], tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target, training):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att, training = training))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out, training = training) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training = training))
        return ffn_out_norm

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # Expandir a dimensão dos inputs se estiver faltando a dimensão embed_dim
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)
        
        # Validar a forma dos inputs
        tf.debugging.assert_rank(inputs, 3, message="Input tensor must have rank 3 (batch_size, seq_len, embed_dim)")
        
        # Atenção multi-cabeça
        attn_output = self.att(query=inputs, value=inputs, key=inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class DataProcessor:
    def __init__(self, input_filename, output_filename):
        self.csv_input_path = os.path.join('E:\libria\data', input_filename)
        self.csv_output_path = os.path.join('E:\libria\data', output_filename)

    def load_or_process_data(self):
        # Carregar o Dataset do CSV
        dataset_df = pd.read_csv(self.csv_input_path)
        print(f"Full dataset shape is {dataset_df.shape}")

        # Verificar se já existe um CSV com os landmarks processados
        if os.path.exists(self.csv_output_path):
            print("Carregando landmarks previamente processados...")
            landmarks_df = pd.read_csv(self.csv_output_path)
            labels = landmarks_df['label'].values
            landmark_features = landmarks_df.drop(columns=['label']).values
        else:
            labels, landmark_features = self.process_images(dataset_df)
            self.save_landmarks(labels, landmark_features)
        
        return labels, landmark_features

    def process_images(self, dataset_df):
        print("Processando imagens com MediaPipe para extrair landmarks...")
        # Separar labels e pixels
        labels = dataset_df['label'].values
        pixels = dataset_df.drop(columns=['label']).values

        # Redimensionar os pixels para o formato esperado pelo MediaPipe (imagens 28x28)
        pixels_reshaped = pixels.reshape(-1, 28, 28)

        # Inicializar MediaPipe para detecção de landmarks
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        landmark_features = []

        # Processar cada imagem com MediaPipe
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            for image in tqdm(pixels_reshaped, desc="Processando imagens com MediaPipe"):
                # Converter imagem para RGB
                image_rgb = np.stack([image] * 3, axis=-1).astype(np.uint8)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extrair landmarks como features
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        landmark_features.append(landmarks)
                else:
                    # Caso não seja detectada uma mão, adicionar um vetor de zeros
                    landmark_features.append([0] * 63)

        return labels, np.array(landmark_features)

    def save_landmarks(self, labels, landmark_features):
        # Converter os landmarks para um array NumPy e salvar em CSV
        landmarks_df = pd.DataFrame(landmark_features, columns=[f'landmark_{i}' for i in range(63)])
        landmarks_df.insert(0, 'label', labels)
        landmarks_df.to_csv(self.csv_output_path, index=False)
        print(f"Landmarks salvos em {self.csv_output_path}")

class DataLoader:
    def __init__(self, labels, images, landmark_features):
        self.labels = labels
        self.images = images
        self.landmark_features = landmark_features

    def prepare_data(self):
        # Converter labels para valores numéricos usando LabelEncoder
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(self.labels)
        num_classes = len(label_encoder.classes_)

        # One-hot encoding dos rótulos
        labels_encoded = keras.utils.to_categorical(labels_encoded, num_classes=num_classes)

        # Dividir o dataset em treino e validação
        X_train_images, X_val_images, X_train_landmarks, X_val_landmarks, y_train, y_val = train_test_split(
            self.images, self.landmark_features, labels_encoded, test_size=0.2, random_state=42
        )

        # Replicar os rótulos para 8 passos no tempo
        y_train = np.repeat(y_train[:, np.newaxis, :], 8, axis=1)  # Forma: (batch_size, 8, num_classes)
        y_val = np.repeat(y_val[:, np.newaxis, :], 8, axis=1)      # Forma: (batch_size, 8, num_classes)

        # Converter para tensores do TensorFlow
        X_train_images = tf.convert_to_tensor(X_train_images, dtype=tf.float32)
        X_train_landmarks = tf.convert_to_tensor(X_train_landmarks, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        X_val_images = tf.convert_to_tensor(X_val_images, dtype=tf.float32)
        X_val_landmarks = tf.convert_to_tensor(X_val_landmarks, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

        # Normalizar os landmarks e imagens (opcional)
        X_train_landmarks = (X_train_landmarks - tf.reduce_mean(X_train_landmarks)) / tf.math.reduce_std(X_train_landmarks)
        X_val_landmarks = (X_val_landmarks - tf.reduce_mean(X_val_landmarks)) / tf.math.reduce_std(X_val_landmarks)

        X_train_images = (X_train_images - tf.reduce_mean(X_train_images)) / tf.math.reduce_std(X_train_images)
        X_val_images = (X_val_images - tf.reduce_mean(X_val_images)) / tf.math.reduce_std(X_val_images)

        # Criar datasets do TensorFlow
        train_ds = tf.data.Dataset.from_tensor_slices(
            ({"image_input": X_train_images, "landmark_input": X_train_landmarks}, y_train)
        ).shuffle(1000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            ({"image_input": X_val_images, "landmark_input": X_val_landmarks}, y_val)
        ).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds, num_classes

class GestureResNet:
    def __init__(self, image_input_shape: tuple, gesture_features_dim: int = 42, num_classes_units: int = None, num_blocks: list[int] = [2, 2, 2, 2]):
        """
        Inicializa a arquitetura GestureResNet.
        """
        self.num_classes_units = num_classes_units
        self.gesture_features_dim = gesture_features_dim
        self.model = self.build_model(image_input_shape, num_blocks)

    def residual_block(self, x, filters: int = 75, kernel_size: int = 3, stride: int = 1):
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

        model = Model(inputs, x)
        return model

class MultiModalGestureNet:
    def __init__(self, image_shape=(28, 28, 1), gesture_features_dim=63, num_blocks=[2, 2, 2, 2], num_classes=None):
        self.image_shape = image_shape
        self.gesture_features_dim = gesture_features_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.model = None

    def load_data(self):
        data_processor = DataProcessor(input_filename='signals.csv', output_filename='landmarks.csv')
        labels, landmark_features = data_processor.load_or_process_data()

        # Carregar as imagens do arquivo de entrada
        images = pd.read_csv(data_processor.csv_input_path).drop(columns=['label']).values.astype('float32')
        images = images.reshape((-1, 28, 28, 1))  # Assumindo que as imagens são 28x28x1

        data_loader = DataLoader(labels, images, landmark_features)
        train_ds, val_ds, num_classes = data_loader.prepare_data()

        self.num_classes = num_classes
        return train_ds, val_ds

    def build_model(self):
        # Entrada de imagem para extração de características com ResNet
        image_input = layers.Input(shape=self.image_shape, name="image_input")
        self.resnet = GestureResNet(self.image_shape, num_blocks=self.num_blocks)
        image_features = self.resnet.model(image_input)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Dropout(0.03)(image_features)

        # Entrada dos landmarks
        landmark_input = layers.Input(shape=(self.gesture_features_dim,), name="landmark_input")
        landmark_features = layers.Dense(64, activation='relu')(landmark_input)
        landmark_features = layers.BatchNormalization()(landmark_features)
        landmark_features = layers.Dropout(0.3)(landmark_features)

        # Concatenar as saídas das duas redes
        concatenated = layers.Concatenate()([image_features, landmark_features])

        # Resto da rede
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))(concatenated)
        x = layers.RepeatVector(8)(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
        attention = layers.Attention()([x, x])
        x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(attention)
        decoder = layers.GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
        output_landmarks = layers.TimeDistributed(layers.Dense(self.num_classes, activation='linear'), name="landmark_output")(decoder)
        output_class = layers.TimeDistributed(layers.Dense(self.num_classes, activation='softmax'), name="class_output")(decoder)

        # Definindo o modelo com todas as entradas
        self.model = Model(inputs=[image_input, landmark_input], outputs=[output_landmarks, output_class])

        # Compilando o modelo
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss=['mean_squared_error', 'categorical_crossentropy'],
            metrics=[
                keras.metrics.Precision(),
                keras.metrics.Recall()
            ]
        )
