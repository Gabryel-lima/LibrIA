# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json)
from src.utils.plots import plot_training_history
import keras
from src.core.MultiModalGestureNet import MultiModalGestureNet
import cv2 as cv
from sklearn.utils import shuffle

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
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Rotação aleatória (menos agressiva)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32))

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
        batch_size = 64

        # Dataset de treino com aumento de dados
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_img, X_train_gesture_features, X_train_pixels), y_train_seq))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train_img))
        train_dataset = train_dataset.map(lambda features, label: apply_augment(features, label), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Dataset de validação
        val_dataset = tf.data.Dataset.from_tensor_slices(((X_test_img, X_test_gesture_features, X_test_pixels), y_test_seq))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Compilação do modelo com hiperparâmetros do otimizador ajustados
        libria.build_model()
        libria.model.summary()

        optimizer = keras.optimizers.AdamW(learning_rate=0.0005)
        libria.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Configuração de callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Monitorar a precisão de validação
            mode='max',  # Parar quando alcançar o valor máximo
            patience=20,  # Número de épocas para esperar antes de parar, sem melhoria
            restore_best_weights=True  # Restaurar os melhores pesos alcançados
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=15,  # Reduzir a taxa de aprendizado mais cedo para otimizar o aprendizado
            verbose=1,
            factor=0.5,  # Fator de redução da taxa de aprendizado
            min_lr=1e-6,
            cooldown=3
        )

        checkpoint_cb = keras.callbacks.ModelCheckpoint('./model/best_model.keras', save_best_only=True)
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

        # Treinamento do modelo com os callbacks ajustados
        history = libria.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=200,
            callbacks=[early_stopping, reduce_lr, checkpoint_cb, tensorboard_cb]
        )

        # Plot do histórico de treinamento
        plot_training_history(history)

        # Avaliação do modelo
        test_loss, test_accuracy = libria.model.evaluate(val_dataset)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

        # Salvar o modelo e os resultados
        libria.model.save('./model/LibriaCombinedModel.keras')
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

def find_last_conv_layer(model):
    """Encontra a última camada convolucional no modelo."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

def generate_gradcam(submodel, processed_image, class_index):
    """
    Gera o Grad-CAM de uma imagem processada para uma classe específica usando apenas o submodelo de imagem.
    """
    last_conv_layer_name = find_last_conv_layer(submodel)
    grad_model = keras.models.Model(inputs=submodel.inputs, outputs=[submodel.get_layer(last_conv_layer_name).output, submodel.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_image)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i in range(pooled_grads.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

def display_gradcam(frame, heatmap, alpha=0.4):
    """Sobrepõe o mapa de calor na imagem original."""
    heatmap = cv.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    superimposed_image = cv.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)
    return superimposed_image

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

def webcam_predictor():
    """
    Função para capturar sinais de mão, detectar landmarks, e visualizar a detecção em tempo real.
    """
    input_shape = (28, 28, 1)
    landmark_dim = 42
    num_classes = 29

    libria = MultiModalGestureNet(image_shape=input_shape, gesture_features_dim=landmark_dim, num_blocks=[2, 2, 2, 2], num_classes=num_classes)
    libria.build_model()
    libria.model.load_weights('./model/LibriaCombinedModel.keras')
    libria.model.summary()

    # Extraia o submodelo ResNet
    resnet_model = libria.resnet.model

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
        processed_image = preprocess_input(frame, target_size=(28, 28))  # Agora `processed_image` tem forma (1, 28, 28, 1)

        # Criar um placeholder para os landmarks de entrada
        dummy_landmarks = np.zeros((1, 42), dtype=np.float32)  # Tamanho de 42, conforme o modelo espera
        dummy_pixels = np.zeros((1, 28, 28, 1), dtype=np.float32)  # Placeholder para os pixels adicionais, conforme necessário

        # Realizar a predição com todas as entradas necessárias
        prediction = libria.model.predict([processed_image, dummy_landmarks, dummy_pixels])

        # Obter a classificação do sinal
        pred_label = np.argmax(prediction[0, -1, :])  # Obter a classe do último frame
        label_text = class_labels.get(pred_label, 'Desconhecido')

        # Gera o Grad-CAM usando apenas o submodelo convolucional
        heatmap = generate_gradcam(resnet_model, processed_image, pred_label)
        superimposed_image = display_gradcam(frame, heatmap)

        # Adicionar a classe prevista na imagem
        cv.putText(superimposed_image, f'Classe: {label_text}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow('Libria - Grad-CAM', superimposed_image)

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
            'train': train_model,
            'cam': webcam_predictor
        }

        choice = input("\nDigite 'train' para treinar o modelo ou 'cam' para abrir a câmera: ").lower()
        
        if choice in actions:
            actions[choice]()
        else:
            print("Opção inválida. Por favor, tente novamente.")

    except Exception as e:
        error_log()
        print(f"Ocorreu um erro: {e}")

def main():
    eval_input()

if __name__ == '__main__':
    main()