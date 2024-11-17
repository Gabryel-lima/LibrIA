# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json, pd)
from src.utils.plots import plot_training_history
import keras
from src.core.GestureNet import Transformer, DataProcessor, DataLoader
import cv2 as cv
from sklearn.utils import shuffle
import os
from keras.src.utils import plot_model
from src.utils.gradients import value_gradient
import mediapipe as mp

# Configurando o TensorFlow para usar todos os threads disponíveis
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

def error_log():
    """Função para registrar o erro em um arquivo log."""
    with open('error_log.txt', 'w') as f:
        f.write('An exception occurred:\n')
        f.write(traceback.format_exc())
    print('An exception occurred. Check error_log.txt for details.')

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
    """Função principal para o treinamento do modelo."""
    try:
        # Carregar os dados de treino e validação dos dois arquivos CSVs (signals.csv e landmarks.csv)
        data_processor = DataProcessor('signals.csv', 'landmarks.csv')
        labels, signals, landmark_features = data_processor.load_or_process_data()
        data_loader = DataLoader(labels, signals, landmark_features)
        train_dataset, val_dataset, num_classes = data_loader.prepare_data()

        # Inicialização do modelo Transformer
        landmark_dim = 63
        gesture_net = Transformer( # Diminui alguns parâmetros pela demora das epochs
            num_hid=32,
            num_head=1,
            num_feed_forward=32,
            source_maxlen=100,
            target_maxlen=100,
            num_layers_enc=2,
            num_layers_dec=1,
            num_classes=num_classes,
        )

        # Construir o modelo especificando a forma da entrada
        gesture_net.build(input_shape=[(None, 100, landmark_dim), (None, 100)])

        # Compilar o modelo com hiperparâmetros do otimizador ajustados
        gesture_net.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),  # clipnorm limita o valor dos gradientes
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[ 'accuracy'
                #keras.metrics.Accuracy()
            ]
        )

        # Sumário do modelo e plot do modelo
        gesture_net.summary()

        # plot_model(gesture_net, to_file='model_structure.png', dpi=200, rankdir='TB',
        #            show_shapes=True, show_layer_names=True, show_trainable=True, show_layer_activations=True)

        # Salvar o `build_config` do modelo para análise
        build_config = gesture_net.get_config()
        with open('model_build_config.json', 'w') as config_file:
            json.dump(build_config, config_file, indent=4)

        # Configuração de callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=10, restore_best_weights=True, min_delta=0.001
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=1e-6, cooldown=2
        )

        lr_scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: float(lr) if epoch < 10 else float(lr * tf.math.exp(-0.1).numpy())
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath='./model/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1
        )

        csv_logger = keras.callbacks.CSVLogger('training_log.csv')
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True, write_steps_per_second=True)
        terminate_nan = keras.callbacks.TerminateOnNaN()

        # Adicionando os callbacks à lista
        callbacks = [
            early_stopping,
            reduce_lr,
            checkpoint,
            csv_logger,
            tensorboard_callback,
            terminate_nan,
            lr_scheduler
        ]

        # Chamando o fit com os callbacks
        history = gesture_net.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            callbacks=callbacks,
            verbose=1
        )

        # Salvando o modelo final
        gesture_net.save('./model/GestureNet.keras')

        # Avaliação final do modelo
        evaluation_results = gesture_net.evaluate(val_dataset)
        print(f'Avaliação do modelo retornou: {evaluation_results}')

        # Salvando o histórico do treinamento
        history_results = {key: value for key, value in history.history.items()}

        # Salvando os resultados do treinamento e da avaliação em um arquivo JSON
        final_results = {
            "evaluation": {
                'test_loss': evaluation_results[0],
                'accuracy': evaluation_results[1]
            },
            "training_history": history_results
        }

        with open('results.json', 'w') as f:
            json.dump(final_results, f, indent=4)

    except Exception as e:
        error_log()
        print(f'Error: {e}')

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
    Função para capturar sinais de mão, detectar landmarks, prever sinais de mão e visualizar a detecção em tempo real.
    """
    input_shape = (28, 28, 1)
    landmark_dim = 63
    num_classes = 29

    # Inicializar e carregar o modelo Transformer
    model = Transformer(
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=num_classes,
    )
    model.build(input_shape=[(None, 100, landmark_dim), (None, 100)])  # Ajuste necessário para o modelo ser utilizado corretamente
    model.load_weights('./model/GestureNet.keras')

    # Inicializar MediaPipe para captura de landmarks
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Erro: Câmera não disponível. Saindo...')
        return

    class_labels = {i: f'Sinal {chr(65 + i)}' for i in range(num_classes)}

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Erro ao capturar frame. Saindo...')
                break

            # Processar a imagem com MediaPipe para detectar landmarks
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # Pré-processar a imagem de entrada para a rede neural
            processed_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            processed_image = cv.resize(processed_image, (28, 28))
            processed_image = processed_image / 255.0  # Normalizar os pixels para [0, 1]
            processed_image = np.expand_dims(processed_image, axis=-1)  # Adicionar canal
            processed_image = np.expand_dims(processed_image, axis=0)  # Adicionar dimensão do batch

            # Placeholder para características dos gestos (landmarks) - aqui será usado como entrada para o modelo
            gesture_features = np.zeros((1, landmark_dim), dtype=np.float32)

            # Se os landmarks foram detectados, preencher gesture_features com os valores
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    gesture_features = np.array(landmarks, dtype=np.float32).reshape(1, -1)  # Forma (1, 63)

                    # Desenhar os landmarks na imagem original
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Realizar a predição com todas as entradas necessárias
            inputs = [gesture_features, np.ones((1, 100))]  # Usando apenas landmarks como entrada e sequência fictícia
            predictions = model(inputs, training=False)

            # Obter a classificação do sinal
            pred_label = np.argmax(predictions[0])  # Classe prevista
            label_text = class_labels.get(pred_label, 'Desconhecido')

            # Exibir resultados
            cv.putText(frame, f'Classe: {label_text}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
            cv.imshow('libria_net - Gesture Detector', frame)

            # Fechar ao pressionar 'q'
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
        error_log()
        print(f"Ocorreu um erro: {e}")

def main():
    eval_input()

if __name__ == '__main__':
    main()
