# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json, pd)
from src.utils.plots import plot_training_history
from src.utils.error_log import error_log
import keras
from keras import Model
from src.core.GestureNet import Transformer, DataProcessor, DataLoader
import cv2 as cv
from sklearn.utils import shuffle
import os
from keras.src.utils import plot_model
from src.utils.gradients import value_gradient
import mediapipe as mp

# Configurando o TensorFlow para usar todos os threads disponíveis
# tf.config.threading.set_intra_op_parallelism_threads(0)
# tf.config.threading.set_inter_op_parallelism_threads(0)

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

    # Garantir que a imagem tenha tamanho consistente ao final (28, 28)
    image = tf.image.resize(image, [28, 28])

    return image

def apply_augment(features, label):
    """Função para aplicar augment apenas na imagem, mantendo landmarks e labels intactos."""
    # Extrair a imagem e landmarks
    image, landmarks = features  # (imagem, landmarks)

    # Aplicar a função de aumento de dados apenas à imagem
    augmented_image = augment(image)

    # Retornar a tupla com a imagem aumentada e os landmarks inalterados, junto com o label
    return (augmented_image, landmarks), label

def train_model():
    try:
        # Preparação dos dados
        data_loader = DataLoader('asl_signals.csv', 'random_hands.csv')
        train_ds, val_ds, num_classes = data_loader.prepare_data()

        # Aplicar aumento de dados ao conjunto de treinamento, se necessário
        #train_ds = train_ds.map(apply_augment, num_parallel_calls=tf.data.AUTOTUNE)

        # Obter um batch de dados para verificar as formas
        for batch in train_ds.take(1):
            inputs, labels = batch
            images, landmarks = inputs
            print(f"[DEBUG] Source shape: {images.shape}")
            print(f"[DEBUG] Target landmarks shape: {landmarks.shape}")
            print(f"[DEBUG] Labels shape: {labels.shape}")
            break

        # Model Gesture_Net_Transformer (número de classes = 26)
        gesture_net = Transformer(num_classes=num_classes)

        # Definir entradas
        images_input = keras.Input(shape=images.shape[1:], batch_size=32)
        landmarks_input = keras.Input(shape=landmarks.shape[1:], batch_size=32)

        # Definir saída do modelo
        outputs = gesture_net((images_input, landmarks_input))

        # Modelo funcional keras
        functional_model = Model(inputs=[images_input, landmarks_input], outputs=outputs)

        # Summary
        functional_model.summary()

        # Compilação
        functional_model.compile(
            optimizer=gesture_net.optimizer,
            loss=gesture_net.compiled_loss,
            metrics=gesture_net.metrics
        )

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, min_lr=1e-6)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath='./model/best_model.keras', monitor='val_loss', save_best_only=True)
        csv_logger = keras.callbacks.CSVLogger('training_log.csv')
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

        callbacks = [early_stopping, reduce_lr, checkpoint, csv_logger, tensorboard_callback]

        # Treinar o modelo funcional
        history = functional_model.fit(
            train_ds.prefetch(tf.data.AUTOTUNE),
            validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
            epochs=30,
            callbacks=callbacks
        )

        # Salvar os melhores pesos
        functional_model.save('./model/GestureNet.keras')
        evaluation_results = functional_model.evaluate(val_ds)
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

def apply_gradcam(model, image, layer_name):
    """
    Applies Grad-CAM to visualize model attention.
    Args:
        model: The Keras model to use for inference.
        image: The input image, preprocessed as expected by the model.
        layer_name: The name of the last convolutional layer.
    Returns:
        heatmap: Heatmap of the Grad-CAM.
    """
    grad_model = keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv.resize(heatmap.numpy(), (image.shape[2], image.shape[1]))

    return heatmap

def webcam_predictor():
    """
    Function to capture hand signals, detect landmarks, predict hand signs, and visualize detection in real-time.
    """
    # Initialize and load the Transformer model
    model = Transformer()
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
        25: 'Z'
    }

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
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

                    # Normalize landmarks (assuming they are between 0 and 1)
                    gesture_features = (gesture_features - 0.5) / 0.5  # Normalize to range [-1, 1]

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

def debug_model():
    # Diretório com imagens de teste
    test_images_dir = 'E:\\libria\\data\\asl_hands\\ASL_Alphabet_Dataset\\asl_alphabet_test'  # Substitua pelo caminho onde estão as imagens estáticas

    # Inicializar o modelo
    model = Transformer()
    model.load_weights('./model/GestureNet.keras')

    # Inicializar MediaPipe para detecção de landmarks
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Mapear índices de classes para rótulos
    class_labels = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
        5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
        25: 'Z'
    }

    # Carregar e processar cada imagem do diretório
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        for image_name in os.listdir(test_images_dir):
            # Carregar a imagem
            image_path = os.path.join(test_images_dir, image_name)
            image = cv.imread(image_path)
            if image is None:
                print(f"Erro ao carregar a imagem: {image_name}")
                continue

            # Processar a imagem com MediaPipe para detectar landmarks
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            result = hands.process(rgb_image)

            # Preprocessar a imagem para o modelo de classificação
            processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            processed_image = cv.resize(processed_image, (28, 28))
            processed_image = processed_image / 255.0  # Normalizar pixels para [0, 1]
            processed_image = np.expand_dims(processed_image, axis=-1)  # Adicionar canal
            processed_image = np.expand_dims(processed_image, axis=0)  # Adicionar dimensão do batch

            # Inicializar gesture_features
            gesture_features = np.zeros((1, 63), dtype=np.float32)  # Assumindo 21 landmarks com coordenadas x, y, z

            # Se os landmarks forem detectados, preencher gesture_features com os valores
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    gesture_features = np.array(landmarks, dtype=np.float32).reshape(1, -1)  # Forma (1, 63)

                    # Desenhar os landmarks na imagem original
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Expandir gesture_features para coincidir com a forma esperada de entrada
            gesture_features = np.expand_dims(gesture_features, axis=1)  # Forma (1, 1, 63)

            # Fazer a predição
            inputs = (processed_image, gesture_features)
            predictions = model(inputs, training=False)

            # Aplicar softmax para obter probabilidades
            pred_probabilities = tf.nn.softmax(predictions[0]).numpy()
            pred_label = np.argmax(pred_probabilities)
            confidence = pred_probabilities[pred_label]

            # Mostrar o rótulo da predição
            label_text = f'{class_labels.get(pred_label, "Unknown")} ({confidence * 100:.2f}%)'
            cv.putText(image, f'Class: {label_text}', (15, 30), cv.FONT_HERSHEY_COMPLEX, 0.7, (30, 30, 30), 2)

            # Mostrar a imagem com o rótulo e a confiança
            cv.imshow('libria_net - Static Image Gesture Detector', image)

            # Esperar até o usuário pressionar qualquer tecla para fechar a janela e passar para a próxima imagem
            cv.waitKey(0)

    cv.destroyAllWindows()

# Função de entrada para iniciar o treinamento ou visualização da câmera
def eval_input():
    """Garante o input corretamente"""
    try:
        actions = {
            '1': train_model,
            '2': webcam_predictor,
            '3': debug_model
        }

        choice = str(input("\nDigite '1' para treinar o modelo, '2' para abrir a câmera ou '3' para o debug: "))
        
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
