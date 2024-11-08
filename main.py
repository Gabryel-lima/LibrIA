# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json)
from src.utils.plots import plot_training_history
import keras
from src.core.MultiModalGestureNet import MultiModalGestureNet
import cv2 as cv
from sklearn.utils import shuffle
import os
from keras.src.utils import plot_model
from src.utils.gradients import value_gradient

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
    image, landmarks, pixels = features

    # Aplicar a função de aumento de dados apenas à imagem
    augmented_image = augment(image)

    # Retornar a tupla (imagem aumentada, landmarks, pixels) junto com o label
    return (augmented_image, landmarks, pixels), label

def train_model():
    """Função principal para o treinamento do modelo."""
    try:
        # Inicialização da classe MultiModalGestureNet e carregamento de dados
        libria = MultiModalGestureNet(image_shape=(28, 28, 1), gesture_features_dim=42, num_blocks=[2, 2, 2, 2])

        # Carregar os dados de treino e teste usando a nova abordagem compacta
        X_train, X_test, y_train_seq, y_test_seq = libria.load_data()

        # Descompactar os dados de treino e teste
        X_train_img, X_train_gesture_features, X_train_pixels = X_train
        X_test_img, X_test_gesture_features, X_test_pixels = X_test

        # Embaralhar os dados de treino
        X_train_img, X_train_gesture_features, X_train_pixels, y_train_seq = shuffle(
            X_train_img, X_train_gesture_features, X_train_pixels, y_train_seq, random_state=42
        )

        # Criar o dataset do TensorFlow
        batch_size = 128

        # Dataset de treino com aumento de dados
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_img, X_train_gesture_features, X_train_pixels), y_train_seq))
        train_dataset = train_dataset.shuffle(len(X_train))
        train_dataset = train_dataset.map(lambda features, label: apply_augment(features, label), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Dataset de validação
        val_dataset = tf.data.Dataset.from_tensor_slices(((X_test_img, X_test_gesture_features, X_test_pixels), y_test_seq))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Compilação do modelo com hiperparâmetros do otimizador ajustados
        libria.build_model()

        # Sumário da rede mãe e plot do modelo
        libria.model.summary()

        # Sumário do submodelo
        libria.resnet.model.summary()

        plot_model(libria.model, to_file='model_structure.png', dpi=200, rankdir='TB',
                   show_shapes=True, show_layer_names=True, show_trainable=True, show_layer_activations=True)

        # Configuração de callbacks melhorada
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Focar na perda de validação
            mode='min',  # Parar quando não houver mais diminuição
            patience=10,  # Número de épocas para esperar antes de parar, sem melhoria
            restore_best_weights=True,
            min_delta=0.001
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # Monitorar a perda de validação para reduzir a LR
            patience=3,  # Reduzir a taxa de aprendizado mais cedo para otimizar o aprendizado
            verbose=1,
            factor=0.2,  # Fator de redução da taxa de aprendizado
            min_lr=1e-6,
            cooldown=2
        )

        lr_scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: float(lr) if epoch < 10 else float(lr * tf.math.exp(-0.1).numpy())
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath='./model/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
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
            lr_scheduler  # Incluindo o agendador de aprendizado personalizado
        ]

        # Chamando o fit com os callbacks
        history = libria.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=1
        )

        # Salvando o modelo
        libria.model.save('./model/LibriaCombinedModel.keras')

        # Plot do histórico de treinamento
        plot_training_history(history)

        # Avaliação do modelo
        test_loss, test_accuracy = libria.model.evaluate(val_dataset)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

        # Resultados
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'training_loss': history.history['loss'],
            'validation_loss': history.history['val_loss'],
            'training_accuracy': history.history['accuracy'],
            'validation_accuracy': history.history['val_accuracy']
        }

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)

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

def webcam_predictor():
    """
    Função para capturar sinais de mão, detectar landmarks, e visualizar a detecção em tempo real.
    """
    input_shape = (28, 28, 1)
    landmark_dim = 42
    num_classes = 29

    # Inicializar e construir o modelo MultiModalGestureNet
    libria = MultiModalGestureNet(image_shape=input_shape, gesture_features_dim=landmark_dim, num_blocks=[2, 2, 2, 2], num_classes=num_classes)
    libria.build_model()
    libria.model.load_weights('./model/LibriaCombinedModel.keras')
    libria.model.summary()
    libria.resnet.model.summary()

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Erro: Câmera não disponível. Saindo...')
        return

    class_labels = {i: f'Sinal {chr(65 + i)}' for i in range(num_classes)}

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Erro ao capturar frame. Saindo...')
            break

        # Pré-processar a imagem de entrada para a rede neural
        processed_image = preprocess_input(frame, target_size=(28, 28))

        # Placeholder para características dos gestos (landmarks) com shape (1, 42)
        gesture_features = np.zeros((1, landmark_dim), dtype=np.float32)

        # Placeholder para os pixels adicionais (imagem também redimensionada)
        additional_pixels = processed_image.copy()

        # Realizar a predição com todas as entradas necessárias
        inputs: list = [processed_image, gesture_features, additional_pixels]
        pred_class = libria.model.predict(inputs)

        # Obter a classificação do sinal
        pred_label = np.argmax(pred_class[0, -1, :])  # Classe do último frame
        if pred_label >= num_classes:
            label_text = 'Desconhecido'
        else:
            label_text = class_labels.get(pred_label, 'Desconhecido')

        # Gera o Grad-CAM usando apenas a entrada da imagem
        heatmap = generate_gradcam(libria.resnet.model, processed_image, pred_label)
        superimposed_image = display_gradcam(frame, heatmap)

        # Adicionar a classe prevista na imagem
        cv.putText(superimposed_image, f'Classe: {label_text}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)

        # Mostrar a imagem com a superposição do Grad-CAM e a classe detectada
        cv.imshow('Libria - Grad-CAM com Landmarks', superimposed_image)

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
