import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras import layers, Model
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np


class DataProcessor:
    def __init__(self, signals_filename, random_hands_filename):
        self.signals_csv_path = os.path.join('E:\\libria\\data', signals_filename)
        self.random_hands_csv_path = os.path.join('E:\\libria\\data', random_hands_filename)
        self.preprocessed = "E:\libria\data\processed_hands_in_landmarks.csv"

    def load_or_process_data(self):
        # Carregar os sinais
        signals_df = pd.read_csv(self.signals_csv_path)
        #hands_df = pd.read_csv(self.random_hands_csv_path)
        labels = signals_df['label'].values
        signals = signals_df.drop(columns=['label']).values

        # Verificar se o CSV de random_hands já existe
        if os.path.exists(self.preprocessed):
            # Carregar os random_hands do CSV
            print("Carregando random_hands do arquivo CSV...")
            random_hands_df = pd.read_csv(self.preprocessed)
            hand_features = random_hands_df.drop(columns=['label']).values
        else:
            # Se não existe, processar as imagens para gerar os random_hands
            print("Arquivo de random_hands não encontrado. Processando imagens com MediaPipe...")
            hand_features = self.process_images(signals)

            # Salvar os random_hands processados em um novo arquivo CSV
            self.save_landmarks(labels, hand_features)

        return labels, signals, hand_features

    def process_images(self, signals):
        """Função que converte imagens de sinais para desenhos de landmarks"""
        print("Processando imagens com MediaPipe para extrair landmarks...")
        pixels_reshaped = signals.reshape(-1, 28, 28)

        mp_hands = mp.solutions.hands
        landmark_features = []

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
            for image in tqdm(pixels_reshaped, desc="Processando imagens com MediaPipe"):
                image_rgb = np.stack([image] * 3, axis=-1).astype(np.uint8)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        landmark_features.append(landmarks)
                else:
                    landmark_features.append([0] * 63)

        return np.array(landmark_features)

    def save_landmarks(self, labels, landmark_features):
        # Salvar os landmarks processados em um CSV
        landmarks_df = pd.DataFrame(landmark_features, columns=[f'landmark_{i}' for i in range(63)])
        landmarks_df.insert(0, 'label', labels)
        output_path = os.path.join('E:\\libria\\data', 'processed_hands_in_landmarks.csv')
        landmarks_df.to_csv(output_path, index=False)
        print(f"Landmarks salvos em {output_path}")

class DataLoader(DataProcessor):
    def __init__(self, signals_filename, landmarks_filename):
        super().__init__(signals_filename, landmarks_filename)
        # Usar o DataProcessor para carregar os dados
        self.labels, self.images, self.landmark_features = self.load_or_process_data()

        # Garantir que todos os arrays tenham o mesmo tamanho (sincronizar tamanhos)
        min_size = min(len(self.labels), len(self.images), len(self.landmark_features))
        self.labels = self.labels[:min_size]
        self.images = self.images[:min_size]
        self.landmark_features = self.landmark_features[:min_size]

    def prepare_data(self, max_samples=None):
        label_encoder = LabelEncoder()

        # Definir classes que queremos remover
        classes_to_drop = ['Nothing', 'Space', 'Delete']

        # Filtrar os dados removendo as classes indesejadas
        mask = ~np.isin(self.labels, classes_to_drop)
        
        # Aplicar a máscara para manter a consistência nos tamanhos
        filtered_labels = self.labels[mask]
        filtered_images = self.images[mask]
        filtered_landmarks = self.landmark_features[mask]

        # Garantir que após o filtro os dados ainda estejam sincronizados
        min_size = min(len(filtered_labels), len(filtered_images), len(filtered_landmarks))
        filtered_labels = filtered_labels[:min_size]
        filtered_images = filtered_images[:min_size]
        filtered_landmarks = filtered_landmarks[:min_size]

        # Codificar os rótulos restantes
        labels_encoded = label_encoder.fit_transform(filtered_labels)
        num_classes = len(label_encoder.classes_)  # Deve ser 26 agora

        # Codificar os rótulos como one-hot
        labels_encoded = keras.utils.to_categorical(labels_encoded, num_classes=num_classes)

        # Dividir os dados em conjuntos de treino e validação
        X_train_images, X_val_images, X_train_landmarks, X_val_landmarks, y_train, y_val = train_test_split(
            filtered_images, filtered_landmarks, labels_encoded, test_size=0.2, random_state=42
        )

        # Aplicar SMOTE para aumentar as classes minoritárias
        smote = SMOTE(random_state=42)

        # Combinar as imagens e landmarks para aplicar o SMOTE
        X_train_combined = np.hstack([X_train_images, X_train_landmarks])
        y_train_combined = np.argmax(y_train, axis=1)  # Convertendo rótulos one-hot para inteiros

        # Aplicar o SMOTE
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train_combined)

        # Separar novamente as imagens e landmarks
        num_image_features = X_train_images.shape[1]
        X_train_images_resampled = X_train_resampled[:, :num_image_features]
        X_train_landmarks_resampled = X_train_resampled[:, num_image_features:]
        
        # Re-codificar os rótulos para formato one-hot
        y_train_resampled = keras.utils.to_categorical(y_train_resampled, num_classes=num_classes)

        # Limitar o número de amostras se necessário
        if max_samples is not None:
            X_train_images_resampled = X_train_images_resampled[:max_samples]
            X_train_landmarks_resampled = X_train_landmarks_resampled[:max_samples]
            y_train_resampled = y_train_resampled[:max_samples]

        # Normalizar imagens
        X_train_images_resampled = self.normalize_images(X_train_images_resampled)
        X_val_images = self.normalize_images(X_val_images)

        # Normalizar landmarks
        X_train_landmarks_resampled = self.normalize_landmarks(X_train_landmarks_resampled)
        X_val_landmarks = self.normalize_landmarks(X_val_landmarks)

        # Convertendo os dados para tensores do TensorFlow
        X_train_images_resampled = tf.convert_to_tensor(X_train_images_resampled, dtype=tf.float32)
        X_val_images = tf.convert_to_tensor(X_val_images, dtype=tf.float32)
        X_train_landmarks_resampled = tf.convert_to_tensor(X_train_landmarks_resampled, dtype=tf.float32)
        X_val_landmarks = tf.convert_to_tensor(X_val_landmarks, dtype=tf.float32)

        y_train_resampled = tf.convert_to_tensor(y_train_resampled, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

        # Criar os datasets do TensorFlow com entradas como tuplas
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((X_train_images_resampled, X_train_landmarks_resampled), y_train_resampled)
        ).shuffle(1000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            ((X_val_images, X_val_landmarks), y_val)
        ).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

        print(f"Shape final de X_train_images: {X_train_images_resampled.shape}")
        print(f"Shape final de X_train_landmarks: {X_train_landmarks_resampled.shape}")
        print(f"Shape final de y_train: {y_train_resampled.shape}")

        return train_ds, val_ds, num_classes

    @staticmethod
    def normalize_images(images):
        images = images.astype('float32') / 255.0  # Normalização entre 0 e 1
        images = images.reshape(-1, 28, 28, 1)  # Adicionar canal
        return images

    @staticmethod
    def normalize_landmarks(landmarks):
        landmarks = landmarks.astype('float32')
        mean = np.mean(landmarks, axis=1, keepdims=True)
        std = np.std(landmarks, axis=1, keepdims=True)
        std = np.where(std == 0, 1e-8, std)  # Substituir std = 0 por um pequeno valor epsilon
        return (landmarks - mean) / std

    @staticmethod
    def debug_data(data, name="Data"):
        mean = np.mean(data)
        std = np.std(data)
        is_nan = np.isnan(data).any()
        is_inf = np.isinf(data).any()
        min_val = np.min(data)
        max_val = np.max(data)

        print(f"--- {name} Debug Info ---")
        print(f"Shape: {data.shape}")
        print(f"Mean: {mean}")
        print(f"Std: {std}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Contains NaN: {is_nan}")
        print(f"Contains Inf: {is_inf}")
        print("-------------------------")

class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def build(self, input_shape):
        self.emb.build(input_shape)
        self.pos_emb.build((input_shape[1], self.emb.output_dim))
        super().build(input_shape)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.num_hid = num_hid
        self.maxlen = maxlen
        self.conv1 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")#, kernel_regularizer=keras.regularizers.l2(1e-4))
        self.dropout1 = layers.Dropout(0.2)  # Dropout após primeira conv
        
        self.conv2 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")#, kernel_regularizer=keras.regularizers.l2(1e-4))
        self.dropout2 = layers.Dropout(0.2)  # Dropout após segunda conv

        self.conv3 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")#, kernel_regularizer=keras.regularizers.l2(1e-4))
        self.dropout3 = layers.Dropout(0.2)  # Dropout após terceira conv

        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def build(self, input_shape):
        # Construir as camadas internas
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        seq_len = input_shape[1]
        seq_len = seq_len // (2 ** 3)  # Três convoluções com strides=2
        return (input_shape[0], seq_len, self.num_hid)

    def call(self, x, training=False):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)  # (batch_size, seq_len, 1)

        x = self.conv1(x)
        x = self.dropout1(x, training=training)  # Aplicar dropout

        x = self.conv2(x)
        x = self.dropout2(x, training=training)  # Aplicar dropout

        x = self.conv3(x)
        x = self.dropout3(x, training=training)  # Aplicar dropout

        # Aplicar positional embedding
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        x += positions

        return x

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.3):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=False, mask=None):
        # Multi-head attention
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        final_output = self.layernorm2(out1 + ffn_output)

        return final_output

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(dropout_rate)
        self.enc_dropout = layers.Dropout(dropout_rate)
        self.ffn_dropout = layers.Dropout(dropout_rate)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def build(self, input_shape):
        # input_shape: [enc_output_shape, dec_input_shape]
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # Retorna a forma do dec_input
        return input_shape[1]

    def causal_attention_mask(self, batch_size, seq_len, num_heads, dtype):
        """Gera uma máscara causal para a atenção."""
        # Criação da máscara triangular inferior
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=dtype), -1, 0)
        # Expandir a máscara para suportar múltiplos heads
        mask = tf.reshape(mask, (1, 1, seq_len, seq_len))  # (1, 1, seq_len, seq_len)
        # Replicar a máscara para todos os heads e batches
        mask = tf.tile(mask, [batch_size, num_heads, 1, 1])  # (batch_size, num_heads, seq_len, seq_len)
        return mask

    def call(self, enc_out, target, training=False, mask=None):
        batch_size = tf.shape(target)[0]
        seq_len = tf.shape(target)[1]
        num_heads = self.num_heads
        dtype = target.dtype

        # Gerar a máscara causal
        causal_mask = self.causal_attention_mask(batch_size, seq_len, num_heads, dtype)

        # Auto-atenção com máscara causal
        target_att = self.self_att(
            query=target, value=target, key=target, attention_mask=causal_mask
        )
        target_norm = self.layernorm1(target + self.self_dropout(target_att, training=training))

        # Atenção com codificador usando máscara de padding
        enc_out_att = self.enc_att(
            query=target_norm, value=enc_out, key=enc_out, attention_mask=mask
        )
        enc_out_norm = self.layernorm2(target_norm + self.enc_dropout(enc_out_att, training=training))

        # Feed-forward
        ffn_out = self.ffn(enc_out_norm)
        return self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training=training))

class ImageEmbedding(layers.Layer):
    def __init__(self, num_hid=64):
        super().__init__()
        self.num_hid = num_hid
        self.conv1 = layers.Conv2D(num_hid, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))
        self.pool1 = layers.MaxPooling2D()
        self.dropout1 = layers.Dropout(0.3)  # Dropout após primeira pool

        self.conv2 = layers.Conv2D(num_hid, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))
        self.pool2 = layers.MaxPooling2D()
        self.dropout2 = layers.Dropout(0.3)  # Dropout após segunda pool

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_hid, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))
        self.dropout3 = layers.Dropout(0.3)  # Dropout após camada densa

    def compute_output_shape(self, input_shape):
        # Calcula a saída considerando as camadas Conv2D, MaxPooling e Flatten
        batch_size = input_shape[0]
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        # Primeiro bloco Conv2D -> MaxPooling
        height = height // 2  # MaxPooling reduz pela metade a altura
        width = width // 2

        # Segundo bloco Conv2D -> MaxPooling
        height = height // 2
        width = width // 2

        # A camada Flatten transforma a imagem em um vetor de dimensão `height * width * num_hid`
        output_shape = (batch_size, 1, self.num_hid)

        return tf.TensorShape(output_shape)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)  # Aplicar dropout

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)  # Aplicar dropout

        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout3(x, training=training)  # Aplicar dropout

        x = tf.expand_dims(x, axis=1)  # Expandir dimensão para que possa ser concatenado com o embedding de landmarks
        return x  # Shape final: (batch_size, 1, num_hid)

class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + keras.backend.epsilon()))
    
    def reset_states(self):
        self.precision.reset_state()
        self.recall.reset_state()

class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=1,
        num_feed_forward=64,
        num_layers_enc=1,
        num_layers_dec=1,
        num_classes=26, # len(LabelEncoder.classes_)
        learning_rate=5e-4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_hid = num_hid

        # Embedding layers
        self.image_embedding = ImageEmbedding(num_hid=num_hid)
        self.landmark_embedding = LandmarkEmbedding(num_hid=num_hid)

        # Encoder and Decoder layers
        self.encoder_layers = [
            TransformerEncoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_enc)
        ]
        self.decoder_layers = [
            TransformerDecoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_dec)
        ]

        # Output layer
        self.final_layer = layers.Dense(num_classes)

        # Optimizer and loss
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

        # Metrics
        self.compiled_loss = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
        self.categorical_accuracy = keras.metrics.CategoricalAccuracy(name="accuracy")
        self.precision_metric = keras.metrics.Precision(name="precision")
        self.recall_metric = keras.metrics.Recall(name="recall")
        self.f1_score = F1Score(name="f1_score")

    def build(self, input_shape):
        images_shape, landmarks_shape = input_shape

        # Construir embeddings sem criar tensores diretamente
        self.image_embedding.build(images_shape)
        image_embed_shape = self.image_embedding.compute_output_shape(images_shape)

        self.landmark_embedding.build(landmarks_shape)
        landmark_embed_shape = self.landmark_embedding.compute_output_shape(landmarks_shape)

        # Combinar embeddings
        batch_size = None  # Usar None para indicar tamanho de batch variável
        seq_len = image_embed_shape[1] + landmark_embed_shape[1]
        num_hid = image_embed_shape[2]

        enc_input_shape = tf.TensorShape([batch_size, seq_len, num_hid])

        # Construir encoder layers
        for encoder_layer in self.encoder_layers:
            encoder_layer.build(enc_input_shape)

        # Construir decoder layers
        dec_input_shape = image_embed_shape
        for decoder_layer in self.decoder_layers:
            decoder_layer.build([enc_input_shape, dec_input_shape])
            dec_input_shape = decoder_layer.compute_output_shape([enc_input_shape, dec_input_shape])

        # Construir camada final
        self.final_layer.build(dec_input_shape)

        # Chamar o build da superclasse
        super().build(input_shape)

    def call(self, inputs, training=False):
        images, landmarks = inputs

        if len(images.shape) == 3:
            images = tf.expand_dims(images, axis=-1)

        # Embeddings
        images_emb = self.image_embedding(images)  # (batch_size, 1, num_hid)
        landmarks_emb = self.landmark_embedding(landmarks)  # (batch_size, seq_len, num_hid)

        # Combine embeddings
        enc_input = tf.concat([images_emb, landmarks_emb], axis=1)  # (batch_size, combined_seq_len, num_hid)

        # Encoder
        enc_output = enc_input
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, training=training)

        # Decoder
        dec_input = images_emb  # (batch_size, 1, num_hid)
        dec_output = dec_input
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(enc_output, dec_output, training=training)

        # Final output
        final_output = self.final_layer(dec_output)  # (batch_size, 1, num_classes)
        final_output = tf.squeeze(final_output, axis=1)  # (batch_size, num_classes)

        return final_output

    @property
    def metrics(self):
        return [
            self.compiled_loss,
            self.categorical_accuracy,
            self.precision_metric,
            self.recall_metric,
            self.f1_score
        ]

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def train_step(self, batch):
        inputs, labels = batch
        images, landmarks = inputs

        with tf.GradientTape() as tape:
            preds = self((images, landmarks), training=True)
            loss = self.compiled_loss(labels, preds)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Atualizar métricas
        self.categorical_accuracy.update_state(labels, preds)
        self.precision_metric.update_state(labels, preds)
        self.recall_metric.update_state(labels, preds)
        self.f1_score.update_state(labels, preds)

        return {
            "accuracy": self.categorical_accuracy.result(),
            "precision": self.precision_metric.result(),
            "recall": self.recall_metric.result(),
            "f1_score": self.f1_score.result()
        }

    @tf.function
    def test_step(self, batch):
        inputs, labels = batch
        images, landmarks = inputs

        preds = self((images, landmarks), training=False)
        loss = self.compiled_loss(labels, preds)

        # Atualizar métricas
        self.categorical_accuracy.update_state(labels, preds)
        self.precision_metric.update_state(labels, preds)
        self.recall_metric.update_state(labels, preds)
        self.f1_score.update_state(labels, preds)

        return {
            "accuracy": self.categorical_accuracy.result(),
            "precision": self.precision_metric.result(),
            "recall": self.recall_metric.result(),
            "f1_score": self.f1_score.result()
        }
    
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from collections import Counter

    # Caminhos dos arquivos CSV
    signals_csv_path = "asl_signals.csv"
    landmarks_csv_path = "random_hands.csv"

    # Preparação dos dados
    data_loader = DataLoader(signals_csv_path, landmarks_csv_path)
    train_ds, val_ds, num_classes = data_loader.prepare_data()

    # Função auxiliar para contar classes nos datasets do TensorFlow
    def count_classes_from_dataset(dataset):
        all_labels = []
        for _, labels in dataset:
            all_labels.extend(tf.argmax(labels, axis=1).numpy())
        return Counter(all_labels)

    # Contar as classes no conjunto de treino e validação
    train_class_counts = count_classes_from_dataset(train_ds)
    val_class_counts = count_classes_from_dataset(val_ds)

    # Plotar a distribuição das classes no conjunto de treino
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.bar(train_class_counts.keys(), train_class_counts.values(), color='skyblue')
    plt.title('Distribuição das Classes - Conjunto de Treino')
    plt.xlabel('Classes')
    plt.ylabel('Número de Amostras')
    plt.xticks(ticks=np.arange(num_classes), labels=np.arange(num_classes), rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)

    # Plotar a distribuição das classes no conjunto de validação
    plt.subplot(1, 2, 2)
    plt.bar(val_class_counts.keys(), val_class_counts.values(), color='lightgreen')
    plt.title('Distribuição das Classes - Conjunto de Validação')
    plt.xlabel('Classes')
    plt.ylabel('Número de Amostras')
    plt.xticks(ticks=np.arange(num_classes), labels=np.arange(num_classes), rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.show()


