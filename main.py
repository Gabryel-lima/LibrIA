# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json, pd)
from src.utils.plots import plot_training_history
from src.utils.error_log import error_log
import keras
from src.core.GestureNet import Transformer, DataProcessor, DataLoader
import cv2 as cv
from sklearn.utils import shuffle
import os
from keras.src.utils import plot_model
from src.utils.gradients import value_gradient
import mediapipe as mp

# Configurando o TensorFlow para usar todos os threads disponíveis
# tf.config.threading.set_intra_op_parallelism_threads(12)
# tf.config.threading.set_inter_op_parallelism_threads(12)

# Função de aumento de dados
def augment(image):
    """Função para realizar o aumento de dados com menos ruído e variação"""
    # Ajuste aleatório de brilho e contraste (intervalo reduzido)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.4, upper=1.6)

    # Rotação aleatória (menos agressiva)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Zoom aleatório (intervalo reduzido)
    scales = tf.constant(np.arange(0.9, 1.1, 0.05), dtype=tf.float32)
    scale = tf.random.shuffle(scales)[0]
    new_height = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * scale, tf.int32)
    new_width = tf.cast(tf.cast(tf.shape(image)[1], tf.float32) * scale, tf.int32)
    image = tf.image.resize(image, [new_height, new_width])
    image = tf.image.resize_with_crop_or_pad(image, 28, 28)

    # Adicionar ruído gaussiano (menos intenso)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
    image = tf.add(image, noise)

    # Random flip (horizontal)
    image = tf.image.random_flip_left_right(image)

    # Garantir que a imagem tenha tamanho consistente ao final (28, 28, 1)
    image = tf.image.resize(image, [28, 28])
    image = tf.expand_dims(image, axis=-1)

    return image

def apply_augment(features, label):
    """Função para aplicar augment apenas na imagem, mantendo landmarks e labels intactos."""
    # Extrair as características: imagem, landmarks, pixels
    image = features

    # Aplicar a função de aumento de dados apenas à imagem
    augmented_image = augment(image)

    # Retornar a tupla (imagem aumentada, landmarks, pixels) junto com o label
    return augmented_image, label

def train_model():
    try:
        # Preparação dos dados
        data_processor = DataProcessor('signals.csv', 'landmarks.csv')
        labels, signals, landmark_features = data_processor.load_or_process_data()
        data_loader = DataLoader(labels, signals, landmark_features)
        train_ds, val_ds, num_classes = data_loader.prepare_data()

        # for batch in train_ds.take(1):
        #     inputs, labels = batch
        #     source, target_landmarks = inputs
        #     print(f"[DEBUG] Source shape: {source.shape}")
        #     print(f"[DEBUG] Target landmarks shape: {target_landmarks.shape}")
        #     print(f"[DEBUG] Labels shape: {labels.shape}")

        # Modelo Transformer
        gesture_net = Transformer(
            num_classes=num_classes,
        )

        # Sumário
        gesture_net.summary()

        # Compilação do modelo (sem build explícito)
        gesture_net.compile(optimizer=gesture_net.optimizer)

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath='./model/best_model.keras', monitor='val_loss', save_best_only=True)
        csv_logger = keras.callbacks.CSVLogger('training_log.csv')
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

        callbacks = [early_stopping, reduce_lr, checkpoint, csv_logger, tensorboard_callback]

        # Treinamento
        history = gesture_net.fit(
            train_ds.prefetch(tf.data.AUTOTUNE),
            validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
            epochs=180,
            callbacks=callbacks
        )

        gesture_net.save('./model/GestureNet.keras')
        evaluation_results = gesture_net.evaluate(val_ds)
        print(f"Avaliação final: {evaluation_results}")

        # Salvar histórico e resultados
        with open('results.json', 'w') as f:
            json.dump({
                "evaluation": {"test_loss": evaluation_results[0], "accuracy": evaluation_results[1]},
                "training_history": history.history
            }, f, indent=4)

    except Exception as e:
        error_log(file="error_train")
        print(f"Erro capturado: {e}")

def display_gradcam(frame, heatmap, alpha=0.4):
    """Sobrepõe o mapa de calor na imagem original."""
    heatmap = cv.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    superimposed_image = cv.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)
    return superimposed_image

def preprocess_input(frame, target_size=(28, 28)):
    """Pré-processa o frame para a entrada da rede neural."""
    # Redimensionar a imagem para o tamanho desejado (28x28)
    input_frame = cv.resize(frame, target_size)
    # Converter a imagem para escala de cinza
    input_frame = cv.cvtColor(input_frame, cv.COLOR_BGR2GRAY)
    # Normalizar os valores dos pixels para o intervalo [0, 1]
    input_frame = input_frame.astype("float32") / 255.0
    # Expandir a dimensão para (28, 28, 1)
    input_frame = np.expand_dims(input_frame, axis=-1)
    # Expandir a dimensão para incluir o batch size (1, 28, 28, 1)
    input_frame = np.expand_dims(input_frame, axis=0)
    return input_frame

def find_last_conv_layer(model: keras.Model):
    """Encontra a última camada convolucional no modelo dinamicamente."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer
    raise ValueError("Nenhuma camada convolucional foi encontrada no modelo.")

def generate_gradcam(submodel, image_input, class_index):
    """
    Gera o Grad-CAM de uma imagem processada para uma classe específica usando o submodelo convolucional.
    Args:
        submodel: O submodelo Keras que vai até a última camada convolucional.
        image_input: A entrada de imagem do modelo.
        class_index: Índice da classe alvo para o Grad-CAM.
    Returns:
        heatmap: O mapa de calor gerado pelo Grad-CAM.
    """
    # Encontrar a última camada convolucional
    last_conv_layer = find_last_conv_layer(submodel)

    # Criar um modelo que retorna tanto a saída da última camada convolucional quanto a saída final
    grad_model = keras.models.Model(inputs=submodel.input, outputs=[last_conv_layer.output, submodel.output])

    with tf.GradientTape() as tape:
        # Passar inputs através do grad_model para obter as saídas
        conv_outputs, predictions = grad_model(image_input, training=False)

        # Obter a perda para a classe alvo
        loss = predictions[:, class_index]

    # Calcular o gradiente da perda em relação à saída da última camada convolucional
    grads = tape.gradient(loss, conv_outputs)

    # Reduzir os gradientes para calcular a importância média em cada canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Obter a saída da última camada convolucional
    conv_outputs = conv_outputs[0]

    # Construir o heatmap ponderando a saída dos canais pela importância média
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Aplicar ReLU e normalizar o heatmap
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Retornar o heatmap diretamente, já que é um numpy.ndarray
    return heatmap

import mediapipe as mp

def webcam_predictor():
    """
    Function to capture hand signals, detect landmarks, predict hand signs, and visualize detection in real-time.
    """

    # Initialize and load the Transformer model
    model = Transformer(num_classes=29)
    model.load_weights('./model/GestureNet.keras')

    # Initialize MediaPipe for landmark detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Camera not available. Exiting...')
        return

    # Map class indices to labels
    class_labels = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
        5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
        25: 'Z', 26: 'Space', 27: 'Delete', 28: 'Nothing'
    }

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error capturing frame. Exiting...')
                break

            # Process the image with MediaPipe to detect landmarks
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # Preprocess the image input for the neural network
            processed_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            processed_image = cv.resize(processed_image, (28, 28))
            processed_image = processed_image / 255.0  # Normalize pixels to [0, 1]
            processed_image = np.expand_dims(processed_image, axis=-1)  # Add channel
            processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

            # Initialize gesture_features
            gesture_features = np.zeros((1, 63), dtype=np.float32)  # Assuming 21 landmarks with x, y, z coordinates

            # If landmarks are detected, fill gesture_features with the values
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    gesture_features = np.array(landmarks, dtype=np.float32).reshape(1, -1)  # Shape (1, 63)

                    # Draw the landmarks on the original image
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Expand gesture_features to match expected input shape
                gesture_features = np.expand_dims(gesture_features, axis=1)  # Shape (1, 1, 63)

                # Make prediction
                inputs = (processed_image, gesture_features)
                predictions = model(inputs, training=False)

                # Apply softmax to get probabilities
                pred_probabilities = tf.nn.softmax(predictions[0]).numpy()
                pred_label = np.argmax(pred_probabilities)
                confidence = pred_probabilities[pred_label]

                label_text = f'{class_labels.get(pred_label, "Unknown")} ({confidence * 100:.2f}%)'

                # Display results
                cv.putText(frame, f'Class: {label_text}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
            else:
                # Handle case when no hand is detected
                cv.putText(frame, 'No hand detected', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

            cv.imshow('libria_net - Gesture Detector', frame)

            # Close when 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

# Função de entrada para iniciar o treinamento ou visualização da câmera
def eval_input():
    """Garante o input corretamente"""
    try:
        actions = {
            '1': train_model,
            '2': webcam_predictor
        }

        choice = str(input("\nDigite '1' para treinar o modelo ou '2' para abrir a câmera: "))
        
        if choice in actions:
            actions[choice]()
        else:
            print("Opção inválida. Por favor, tente novamente.")
            return

    except Exception as e:
        error_log(file="error_eval")
        print(f"Ocorreu um erro: {e}")

def main():
    eval_input()

if __name__ == '__main__':
    main()
