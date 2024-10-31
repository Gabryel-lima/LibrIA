# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json)
from src.utils.plots import plot_training_history
import keras
from src.core.libria import Libria
import cv2 as cv

def error_log():
    """Função para registrar o erro em um arquivo log."""
    with open('error_log.txt', 'w') as f:
        f.write('An exception occurred:\n')
        f.write(traceback.format_exc())
    print('An exception occurred. Check error_log.txt for details.')

# Função de aumento de dados
def augment(image, label) -> tuple:
    """Função para realizar o aumento de dados com mais ruído e entropia"""
    # Ajuste aleatório de brilho e contraste
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.1, upper=1.4)
    
    # Rotação aleatória
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Converter para 3 canais para aplicar saturação e matiz
    image_rgb = tf.image.grayscale_to_rgb(image)
    image_rgb = tf.image.random_saturation(image_rgb, lower=0.6, upper=1.6)
    image_rgb = tf.image.random_hue(image_rgb, max_delta=0.2)
    image = tf.image.rgb_to_grayscale(image_rgb)
    
    # Adicionar ruído gaussiano
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    image = tf.add(image, noise)
    
    return image, label

# Função de treinamento do modelo
def train_model():
    """Função principal para o treinamento do modelo."""
    try:
        # Inicialização da classe Libria e carregamento de dados
        libria = Libria(image_shape=(28, 28, 1), landmark_dim=42, num_blocks=[2, 2, 2, 2])
        X_train_img, X_test_img, X_train_landmarks, X_test_landmarks, y_train, y_test = libria.load_data()

        # Construção do modelo
        libria.build_model()
        libria.model.summary()

        # Compilação do modelo
        libria.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Preparar o conjunto de dados de treinamento e validação com duas entradas
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_img, X_train_landmarks), y_train))
        train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices(((X_test_img, X_test_landmarks), y_test)).batch(64)

        # Configuração de callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-6)

        # Treinamento do modelo
        history = libria.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20,
            callbacks=[early_stopping, reduce_lr]
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

def preprocess_input(frame, target_size=(28, 28)):
    """Pré-processa o frame para a entrada da rede neural."""
    input_frame = cv.resize(frame, target_size)
    input_frame = cv.cvtColor(input_frame, cv.COLOR_BGR2GRAY)
    input_frame = input_frame.astype("float32") / 255.0
    input_frame = np.expand_dims(input_frame, axis=[0, -1])
    return input_frame

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

def webcam_predictor():
    """Função para capturar sinais de mão e exibir Grad-CAM usando a webcam."""
    input_shape = (28, 28, 1)
    num_classes = 25
    libria = Libria(image_shape=input_shape, landmark_dim=42, num_blocks=[2, 2, 2, 2], num_classes=num_classes)
    libria.build_model()
    libria.model.load_weights('./model/LibriaCombinedModel.keras')
    libria.model.summary()

    # Extraia o submodelo ResNet
    resnet_model = libria.resnet.model

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Erro: Webcam não disponível. Saindo...')
        return

    class_labels = {i: f'Sinal {chr(65 + i)}' for i in range(num_classes)}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Erro ao capturar frame. Saindo...')
            break
        
        processed_image = preprocess_input(frame)
        #processed_image = np.expand_dims(processed_image, axis=0)  # Expandir para incluir dimensão de batch

        # Supõe que landmarks são zeros neste exemplo; substituir por entrada real, se disponível
        dummy_landmarks = np.zeros((1, 42))  

        prediction = libria.model.predict([processed_image, dummy_landmarks])
        pred_label = np.argmax(prediction)
        label_text = class_labels.get(pred_label, 'Desconhecido')

        # Gera o Grad-CAM usando apenas o submodelo convolucional
        heatmap = generate_gradcam(resnet_model, processed_image, pred_label)
        superimposed_image = display_gradcam(frame, heatmap)
        
        cv.putText(superimposed_image, f'Classe: {label_text}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow('Libria - Grad-CAM', superimposed_image)
        
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
