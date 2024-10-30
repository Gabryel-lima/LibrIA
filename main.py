# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json)
from src.utils.plots import plot_training_history, sample_correct_class
import keras
from src.core.libria import Libria
from src.core.res_net import ResNet
import cv2 as cv
import numpy as np

# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json)
from src.utils.plots import plot_training_history, sample_correct_class

import keras
from src.core.libria import Libria
from src.core.res_net import ResNet
import cv2 as cv
import numpy as np

def error_log():
    with open('error_log.txt', 'w') as f:
        f.write('An exception occurred:\n')
        f.write(traceback.format_exc())
    print('An exception occurred. Check error_log.txt for details.')

# Defina o layer de tradução aleatória fora da função `augment`
random_translation = keras.layers.RandomTranslation(0.2, 0.2)

# Função de aumento de dados
def augment(image, label) -> tuple:
    """Função para realizar o aumento de dados com mais ruído e entropia."""
    # Flip horizontal aleatório
    image = tf.image.random_flip_left_right(image)
    
    # Ajuste aleatório de brilho e contraste
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.1, upper=1.4)
    
    # Rotação aleatória
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Converter para RGB para aplicar saturação e matiz
    image_rgb = tf.image.grayscale_to_rgb(image)
    image_rgb = tf.image.random_saturation(image_rgb, lower=0.6, upper=1.6)
    image_rgb = tf.image.random_hue(image_rgb, max_delta=0.2)
    image = tf.image.rgb_to_grayscale(image_rgb)
    
    # # Adicionar ruído gaussiano
    # noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    # image = tf.add(image, noise)
    
    # # Corte aleatório e redimensionamento
    # cropped_size = int(0.9 * image.shape[0])
    # image = tf.image.random_crop(image, size=[cropped_size, cropped_size, image.shape[2]])
    
    # # Redimensiona novamente para (28, 28, 1)
    # image = tf.image.resize(image, [28, 28])
    
    return image, label

def apply_augmentation(image, label) -> tuple:
    """Função para aplicar aumento de dados."""
    image = random_translation(image)
    return augment(image, label)

def error_log():
    """Função para registrar o erro em um arquivo log."""
    with open('error_log.txt', 'w') as f:
        f.write('An exception occurred:\n')
        f.write(traceback.format_exc())
    print('An exception occurred. Check error_log.txt for details.')

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

def generate_gradcam(model, processed_image, class_index):
    """
    Gera o Grad-CAM de uma imagem processada para uma classe específica.
    Args:
    - model: o modelo keras.
    - processed_image: imagem processada para o modelo.
    - class_index: índice da classe alvo.
    """
    last_conv_layer_name = find_last_conv_layer(model)
    grad_model = keras.models.Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

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
    """
    Sobrepõe o mapa de calor na imagem original.
    Args:
    - frame: imagem original.
    - heatmap: mapa de calor do Grad-CAM.
    - alpha: opacidade do mapa de calor.
    """
    heatmap = cv.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    superimposed_image = cv.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)
    return superimposed_image

def train_model():
    """Função principal para o treinamento do modelo."""
    try:
        libria = Libria(input_shape=(28, 28, 1), num_blocks=[2, 2, 2, 2])
        X_train, X_test, y_train, y_test = libria.load_data()
        libria.ResNet.model.summary()
        libria.ResNet.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

        history = libria.ResNet.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20,
            callbacks=[early_stopping, reduce_lr]
        )
        
        plot_training_history(history)
        
        test_loss, test_accuracy = libria.ResNet.model.evaluate(val_dataset)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        
        libria.ResNet.model.save('./model/LibriaResNet18.keras')
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

def webcam_predictor():
    """Função para capturar sinais de mão e exibir Grad-CAM usando DroidCam ou webcam."""
    input_shape = (28, 28, 1)
    num_classes = 25
    model = ResNet(input_shape=input_shape, num_classes_units=num_classes)
    model.model.load_weights('./model/LibriaResNet18.keras')
    
    class_labels = {i: f'Sinal {chr(65 + i)}' for i in range(num_classes)}
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Erro: Webcam não disponível. Saindo...')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Erro ao capturar frame. Saindo...')
            break
        
        processed_image = preprocess_input(frame)
        prediction = model.model.predict(processed_image)
        pred_label = np.argmax(prediction)
        label_text = class_labels.get(pred_label, 'Desconhecido')
        
        heatmap = generate_gradcam(model.model, processed_image, pred_label)
        superimposed_image = display_gradcam(frame, heatmap)
        
        cv.putText(superimposed_image, f'Classe: {label_text}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow('Libria - Grad-CAM', superimposed_image)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

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
