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

class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = keras.layers.Embedding(num_vocab, num_hid)
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
        self.conv1 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        # Garantir que a entrada esteja no formato correto.
        print("Shape inicial de x:", tf.shape(x))  # Debug do formato inicial de x

        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)  # Expandir para 3D se for 2D (batch_size, seq_len) -> (batch_size, seq_len, 1)
        
        # Aplicar convolução
        x = self.conv1(x)
        print("Shape após conv1:", tf.shape(x))  # Debug após conv1
        tf.debugging.assert_shapes([(x, ('batch_size', 'seq_len', 'num_hid'))], message="Erro após conv1: shape inesperado")

        x = self.conv2(x)
        print("Shape após conv2:", tf.shape(x))  # Debug após conv2
        tf.debugging.assert_shapes([(x, ('batch_size', 'seq_len', 'num_hid'))], message="Erro após conv2: shape inesperado")

        x = self.conv3(x)
        print("Shape após conv3:", tf.shape(x))  # Debug após conv3
        tf.debugging.assert_shapes([(x, ('batch_size', 'seq_len', 'num_hid'))], message="Erro após conv3: shape inesperado")

        return x

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

    def causal_attention_mask(self, batch_size, seq_len, num_heads, dtype):
        """Gera uma máscara causal para a atenção."""
        # Criação da máscara triangular inferior
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=dtype), -1, 0)
        # Expandir a máscara para suportar múltiplos heads
        mask = tf.reshape(mask, (1, 1, seq_len, seq_len))  # (1, 1, seq_len, seq_len)
        # Replicar a máscara para todos os heads e batches
        mask = tf.tile(mask, [batch_size, num_heads, 1, 1])  # (batch_size, num_heads, seq_len, seq_len)
        return mask

    def call(self, enc_out, target, training=False):
        # Capturar as formas de `target` e `enc_out` corretamente
        input_shape = target.get_shape().as_list()
        enc_shape = enc_out.get_shape().as_list()

        # Certifique-se de que todas as dimensões estão bem definidas
        if None in input_shape:
            input_shape = tf.shape(target)
            batch_size = input_shape[0]
            seq_len = input_shape[1] * input_shape[2]  # Achar as dimensões seq_len considerando 7 e 29
        else:
            batch_size = input_shape[0]
            seq_len = input_shape[1] * input_shape[2]

        # Ajustar o `target` para que seja compatível com a atenção
        target = tf.reshape(target, [batch_size, seq_len, input_shape[-1]])  # Flatten as dimensões de seq_len
        print("Shape ajustado de target para atenção:", target.get_shape().as_list())

        # Também capture a dimensão de enc_out
        if None in enc_shape:
            enc_shape = tf.shape(enc_out)

        # Depuração dos shapes para entendimento
        print("Shape de target (após ajuste):", target.get_shape().as_list())
        print("Shape de enc_out:", enc_out.get_shape().as_list())

        # Corrigir a geração da máscara causal
        causal_mask = self.causal_attention_mask(batch_size, seq_len, self.self_att.num_heads, dtype=tf.float32)
        print("Shape da máscara causal:", tf.shape(causal_mask))

        # Aplicar self-attention ao target com a máscara causal
        target_att = self.self_att(
            query=target,
            value=target,
            key=target,
            attention_mask=causal_mask,
            training=training
        )
        print("Shape após self-attention (target_att):", tf.shape(target_att))

        target_norm = self.layernorm1(target + self.self_dropout(target_att, training=training))

        # Aplicar atenção cruzada entre target_norm e enc_out
        enc_out_att = self.enc_att(
            query=target_norm,
            value=enc_out,
            key=enc_out,
            training=training
        )
        enc_out_norm = self.layernorm2(target_norm + self.enc_dropout(enc_out_att, training=training))

        # Aplicar Feedforward
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training=training))

        return ffn_out_norm

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
        landmarks_df.to_csv(self.csv_output_path, index=False)
        print(f"Landmarks salvos em {self.csv_output_path}")

class DataLoader:
    def __init__(self, labels, images, landmark_features):
        self.labels = labels
        self.images = images
        self.landmark_features = landmark_features

    def prepare_data(self):
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(self.labels)
        num_classes = len(label_encoder.classes_)

        # Codificar os rótulos como one-hot
        labels_encoded = keras.utils.to_categorical(labels_encoded, num_classes=num_classes)

        # Dividir os dados em conjuntos de treino e validação
        X_train_images, X_val_images, X_train_landmarks, X_val_landmarks, y_train, y_val = train_test_split(
            self.images, self.landmark_features, labels_encoded, test_size=0.2, random_state=42
        )

        # Convertendo os dados para tensores do TensorFlow
        X_train_images = tf.convert_to_tensor(X_train_images, dtype=tf.float32)
        X_train_landmarks = tf.convert_to_tensor(X_train_landmarks, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        X_val_images = tf.convert_to_tensor(X_val_images, dtype=tf.float32)
        X_val_landmarks = tf.convert_to_tensor(X_val_landmarks, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

        # Normalizando as características das imagens e dos landmarks
        X_train_landmarks = (X_train_landmarks - tf.reduce_mean(X_train_landmarks)) / tf.math.reduce_std(X_train_landmarks)
        X_val_landmarks = (X_val_landmarks - tf.reduce_mean(X_val_landmarks)) / tf.math.reduce_std(X_val_landmarks)

        X_train_images = (X_train_images - tf.reduce_mean(X_train_images)) / tf.math.reduce_std(X_train_images)
        X_val_images = (X_val_images - tf.reduce_mean(X_val_images)) / tf.math.reduce_std(X_val_images)

        # Concatenando as imagens e os landmarks para formar um único tensor de entrada
        X_train = tf.concat([X_train_images, X_train_landmarks], axis=-1)
        X_val = tf.concat([X_val_images, X_val_landmarks], axis=-1)

        # Ajustar a forma do `y_train` para garantir a compatibilidade com `X_train`
        # Aqui, vamos garantir que o comprimento da sequência de `y_train` seja compatível com `X_train`.
        y_train = tf.reshape(y_train, [tf.shape(y_train)[0], -1])  # Ajustar para a forma (batch_size, seq_len)
        y_val = tf.reshape(y_val, [tf.shape(y_val)[0], -1])

        # Verificar compatibilidade entre `X_train` e `y_train`
        if y_train.shape[0] != X_train.shape[0]:
            raise ValueError("O número de amostras em `y_train` não corresponde ao de `X_train`.")

        # Criando os datasets do TensorFlow sem usar dicionário para inputs
        train_ds = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)
        ).shuffle(1000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)
        ).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds, num_classes
    
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

    def call(self, inputs, training):
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)
        
        attn_output = self.att(query=inputs, value=inputs, key=inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=60,
        learning_rate=1e-3
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.Mean(name="edit_dist")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        # Input embeddings para encoder e decoder
        self.enc_input = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        # Encoder stack
        self.encoder_layers = [
            TransformerEncoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_enc)
        ]

        # Decoder layers
        self.decoder_layers = [
            TransformerDecoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_dec)
        ]

        # Classificador final
        self.classifier = layers.Dense(num_classes)
        self.classifier = layers.Dense(num_classes)

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Losses
        self.compiled_loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return self.classifier(x)

    def train_step(self, batch):
        source, target = batch

        # Certifique-se de que o target seja do tipo inteiro
        target = tf.cast(target, tf.int32)

        # Expandir o comprimento do target para coincidir com o comprimento do source
        batch_size = tf.shape(source)[0]
        seq_len = tf.shape(source)[1]  # seq_len do source, deve ser 847 neste caso
        target_seq_len = tf.shape(target)[1]  # Atualmente é 29

        # Ajuste para garantir que target tenha o mesmo comprimento que source
        if target_seq_len != seq_len:
            if target_seq_len < seq_len:
                # Caso o comprimento do target seja menor que o do source, replicamos o target
                repeat_factor = seq_len // target_seq_len
                target = tf.tile(target, [1, repeat_factor])
                # Truncar caso a replicação ultrapasse o comprimento do source
                target = target[:, :seq_len]
            elif target_seq_len > seq_len:
                # Caso o comprimento do target seja maior que o do source, truncamos
                target = target[:, :seq_len]

        # Criação do target_one_hot com a forma correta para coincidir com preds
        target_one_hot = tf.one_hot(target, depth=self.num_classes)  # Deve resultar em (batch_size, seq_len, num_classes)

        # Certifique-se de que a forma de preds seja compatível com target_one_hot
        with tf.GradientTape() as tape:
            preds = self(source, training=True)  # Deve ter a forma (batch_size, seq_len, num_classes)

            # Truncar `preds` para garantir que coincida com `target_one_hot`
            if tf.shape(preds)[1] > tf.shape(target_one_hot)[1]:
                preds = preds[:, :tf.shape(target_one_hot)[1], :]

            # Calcular a perda
            loss = self.compiled_loss(target_one_hot, preds)

        # Calcular gradientes e aplicar as atualizações
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, batch):
        source, target = batch

        # Certifique-se de que o target seja do tipo inteiro
        target = tf.cast(target, tf.int32)

        # Expandir o comprimento do target para coincidir com o comprimento do source
        batch_size = tf.shape(source)[0]
        seq_len = tf.shape(source)[1]  # seq_len do source, deve ser 847 neste caso
        target_seq_len = tf.shape(target)[1]  # Atualmente é 29

        # Ajuste para garantir que target tenha o mesmo comprimento que source
        if target_seq_len != seq_len:
            if target_seq_len < seq_len:
                # Caso o comprimento do target seja menor que o do source, replicamos o target
                repeat_factor = seq_len // target_seq_len
                target = tf.tile(target, [1, repeat_factor])
                # Truncar caso a replicação ultrapasse o comprimento do source
                target = target[:, :seq_len]
            elif target_seq_len > seq_len:
                # Caso o comprimento do target seja maior que o do source, truncamos
                target = target[:, :seq_len]

        # Criação do target_one_hot com a forma correta para coincidir com preds
        target_one_hot = tf.one_hot(target, depth=self.num_classes)  # Deve resultar em (batch_size, seq_len, num_classes)

        # Certifique-se de que a forma de preds seja compatível com target_one_hot
        preds = self(source, training=False)  # Deve ter a forma (batch_size, seq_len, num_classes)

        # Truncar `preds` para garantir que coincida com `target_one_hot`
        if tf.shape(preds)[1] > tf.shape(target_one_hot)[1]:
            preds = preds[:, :tf.shape(target_one_hot)[1], :]

        # Calcular a perda
        loss = self.compiled_loss(target_one_hot, preds)

        return {"loss": loss}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        # Encodar a entrada
        enc = source
        for layer in self.encoder_layers:
            enc = layer(enc, training=False)

        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []

        for _ in range(self.target_maxlen - 1):
            # Passar pela pilha do decoder
            dec_out = dec_input
            for layer in self.decoder_layers:
                dec_out = layer(enc, dec_out, training=False)
            
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)

        return tf.concat(dec_logits, axis=1)

if __name__ == "__main__":
    # Carregar os dados usando o DataProcessor
    signals_filename = "signals.csv"
    landmarks_filename = "landmarks.csv"
    processor = DataProcessor(signals_filename, landmarks_filename)
    labels, signals, landmark_features = processor.load_or_process_data()

    # Preparar o DataLoader e os datasets de treino e validação
    dataloader = DataLoader(labels, signals, landmark_features)
    train_ds, val_ds, num_classes = dataloader.prepare_data()

    # Definir e compilar o modelo Transformer
    transformer_model = Transformer(
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=8,  # Ajustado para o comprimento desejado dos sinais target
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=num_classes,
        learning_rate=1e-3,
    )

    # Realizar o treinamento
    epochs = 10  # Ajuste de acordo com suas necessidades
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Treinamento
        for step, batch in enumerate(train_ds):
            loss = transformer_model.train_step(batch)
            if step % 50 == 0:
                print(f"Step {step}: Loss = {loss['loss'].numpy()}")

        # Validação
        val_loss = []
        for val_step, val_batch in enumerate(val_ds):
            val_result = transformer_model.test_step(val_batch)
            val_loss.append(val_result['loss'].numpy())

        mean_val_loss = np.mean(val_loss)
        print(f"Validation Loss: {mean_val_loss}")

    # Gerar predições para um exemplo (por exemplo, usar um batch de val_ds)
    for batch in val_ds.take(1):
        source, _ = batch
        pred = transformer_model.generate(source, target_start_token_idx=0)
        print("Predição:", pred.numpy())
