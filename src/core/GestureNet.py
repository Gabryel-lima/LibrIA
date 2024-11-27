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

class DataProcessor:
    def __init__(self, signals_filename, landmarks_filename):
        self.signals_csv_path = os.path.join('E:\\libria\\data', signals_filename)
        self.landmarks_csv_path = os.path.join('E:\\libria\\data', landmarks_filename)

    def load_or_process_data(self):
        # Carregar os sinais
        signals_df = pd.read_csv(self.signals_csv_path)
        labels = signals_df['label'].values
        signals = signals_df.drop(columns=['label']).values

        # Carregar os landmarks
        landmarks_df = pd.read_csv(self.landmarks_csv_path)
        landmark_features = landmarks_df.drop(columns=['label']).values

        return labels, signals, landmark_features
    
    def process_images(self, dataset_df):
        print("Processando imagens com MediaPipe para extrair landmarks...")
        labels = dataset_df['label'].values
        pixels = dataset_df.drop(columns=['label']).values
        pixels_reshaped = pixels.reshape(-1, 28, 28)

        mp_hands = mp.solutions.hands
        landmark_features = []

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
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

        return labels, np.array(landmark_features)

    def save_landmarks(self, labels, landmark_features):
        landmarks_df = pd.DataFrame(landmark_features, columns=[f'landmark_{i}' for i in range(63)])
        landmarks_df.insert(0, 'label', labels)
        landmarks_df.to_csv('E:\\data\\', index=False)
        print(f"Landmarks salvos em {'E:\\data\\'}")

class DataLoader:
    def __init__(self, labels, images, landmark_features):
        self.labels = labels
        self.images = images
        self.landmark_features = landmark_features

    def prepare_data(self, max_samples=None):
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(self.labels)
        num_classes = len(label_encoder.classes_)

        # Codificar os rótulos como one-hot
        labels_encoded = keras.utils.to_categorical(labels_encoded, num_classes=num_classes)

        # Dividir os dados em conjuntos de treino e validação
        X_train_images, X_val_images, X_train_landmarks, X_val_landmarks, y_train, y_val = train_test_split(
            self.images, self.landmark_features, labels_encoded, test_size=0.2, random_state=42
        )

        # Limitar o número de amostras se necessário
        if max_samples is not None:
            X_train_images = X_train_images[:max_samples]
            X_train_landmarks = X_train_landmarks[:max_samples]
            y_train = y_train[:max_samples]

        # Verificar os dados brutos
        self.debug_data(X_train_images, name="X_train_images (raw)")
        self.debug_data(X_train_landmarks, name="X_train_landmarks (raw)")

        # Normalizar imagens
        X_train_images = self.normalize_images(X_train_images)
        X_val_images = self.normalize_images(X_val_images)

        # Normalizar landmarks
        X_train_landmarks = self.normalize_landmarks(X_train_landmarks)
        X_val_landmarks = self.normalize_landmarks(X_val_landmarks)

        # Verificar os dados normalizados
        self.debug_data(X_train_images, name="X_train_images (normalized)")
        self.debug_data(X_train_landmarks, name="X_train_landmarks (normalized)")

        # Convertendo os dados para tensores do TensorFlow
        X_train_images = tf.convert_to_tensor(X_train_images, dtype=tf.float32)
        X_val_images = tf.convert_to_tensor(X_val_images, dtype=tf.float32)
        X_train_landmarks = tf.convert_to_tensor(X_train_landmarks, dtype=tf.float32)
        X_val_landmarks = tf.convert_to_tensor(X_val_landmarks, dtype=tf.float32)

        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

        # Criar os datasets do TensorFlow com entradas como tuplas
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((X_train_images, X_train_landmarks), y_train)  # Passando entradas como tuplas
        ).shuffle(1000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            ((X_val_images, X_val_landmarks), y_val)  # Passando entradas como tuplas
        ).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

        print(f"Shape final de X_train_images: {X_train_images.shape}")
        print(f"Shape final de X_train_landmarks: {X_train_landmarks.shape}")
        print(f"Shape final de y_train: {y_train.shape}")

        return train_ds, val_ds, num_classes

    @staticmethod
    def normalize_images(images):
        """Normaliza imagens e adiciona o canal."""
        images = images.astype('float32') / 255.0  # Normalização entre 0 e 1
        images = images.reshape(-1, 28, 28, 1)  # Adicionar canal
        return images

    @staticmethod
    def normalize_landmarks(landmarks):
        """Normaliza landmarks usando z-score normalization, evitando divisão por zero."""
        landmarks = landmarks.astype('float32')
        mean = np.mean(landmarks, axis=1, keepdims=True)
        std = np.std(landmarks, axis=1, keepdims=True)
        std = np.where(std == 0, 1e-8, std)  # Substituir std = 0 por um pequeno valor epsilon
        return (landmarks - mean) / std

    @staticmethod
    def debug_data(data, name="Data"):
        """Função para verificar e depurar dados."""
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
        self.conv1 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv3 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        # Garantir que x tenha três dimensões
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)  # (batch_size, seq_len, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Aplicar positional embedding
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        x += positions

        return x

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
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

    def build(self, input_shape):
        super().build(input_shape)

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
        num_heads = self.self_att.num_heads
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
        self.conv1 = layers.Conv2D(num_hid, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(num_hid, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_hid, activation='relu')

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        # Expandir dimensão para que possa ser concatenado com o embedding de landmarks
        x = tf.expand_dims(x, axis=1)
        return x  # Shape final: (batch_size, 1, num_hid)

class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=29,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_hid = num_hid

        # Metrics
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.categorical_accuracy = keras.metrics.CategoricalAccuracy(name="accuracy")
        self.precision_metric = keras.metrics.Precision(name="precision")
        self.recall_metric = keras.metrics.Recall(name="recall")

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
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.compiled_loss = keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        )

    def build(self, input_shape):
        # Let Keras handle dynamic shapes
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
            self.loss_metric,
            self.categorical_accuracy,
            self.precision_metric,
            self.recall_metric
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
        self.loss_metric.update_state(loss)
        self.categorical_accuracy.update_state(labels, preds)
        self.precision_metric.update_state(labels, preds)
        self.recall_metric.update_state(labels, preds)

        return {
            "loss": self.loss_metric.result(),
            "accuracy": self.categorical_accuracy.result(),
            "precision": self.precision_metric.result(),
            "recall": self.recall_metric.result()
        }

    @tf.function
    def test_step(self, batch):
        inputs, labels = batch
        images, landmarks = inputs

        preds = self((images, landmarks), training=False)
        loss = self.compiled_loss(labels, preds)

        # Atualizar métricas
        self.loss_metric.update_state(loss)
        self.categorical_accuracy.update_state(labels, preds)
        self.precision_metric.update_state(labels, preds)
        self.recall_metric.update_state(labels, preds)

        return {
            "loss": self.loss_metric.result(),
            "accuracy": self.categorical_accuracy.result(),
            "precision": self.precision_metric.result(),
            "recall": self.recall_metric.result()
        }
