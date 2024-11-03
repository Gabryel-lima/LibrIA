import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, layers
from keras.src.utils import plot_model
import cv2
import os

# Simular um modelo ResNet simples para exemplo
def build_example_model(input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = build_example_model()
model.summary()

# 1. Visualização Estrutural da Rede
def visualizar_estrutura_rede(model):
    """Visualiza a arquitetura da rede usando plot_model."""
    plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
    print("Visualização da arquitetura salva como 'model_structure.png'.")

# 2. Visualização de Ativações de Camadas Intermediárias
def visualizar_ativacao_intermediaria(model, sample_input):
    """Visualiza as ativações de camadas intermediárias do modelo."""
    layer_outputs = [layer.output for layer in model.layers[:4]]  # Saída das primeiras 4 camadas
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(sample_input)

    for i, activation in enumerate(activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]
        images_per_row = 8

        n_cols = int(n_features // images_per_row)
        fig, axes = plt.subplots(n_cols, images_per_row, figsize=(12, 12))
        fig.suptitle(f'Ativação da camada {i + 1}', fontsize=16)
        for row in range(n_cols):
            for col in range(images_per_row):
                ax = axes[row, col]
                if col + row * images_per_row < n_features:
                    ax.matshow(activation[0, :, :, col + row * images_per_row], cmap='viridis')
                ax.axis('off')
        plt.show()

# 3. Visualização com Grad-CAM
def generate_gradcam(input_model, image, layer_name):
    """Gera o Grad-CAM de uma imagem processada para uma classe específica."""
    grad_model = Model(inputs=[input_model.inputs], outputs=[input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image]))
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

def visualizar_gradcam(model, sample_input):
    """Visualiza o mapa de ativação (Grad-CAM) para a imagem de exemplo."""
    layer_name = 'conv2d_2'  # Nome da camada convolucional que queremos visualizar
    heatmap = generate_gradcam(model, sample_input[0], layer_name)

    plt.imshow(sample_input[0, :, :, 0], cmap='gray')
    plt.imshow(heatmap, alpha=0.5)
    plt.title("Grad-CAM Visualização")
    plt.show()

if __name__ == '__main__':
    # -----------------------------------------------
    # Exemplos de Input Artificiais
    # -----------------------------------------------

    # 1. Input Artificial - Visualização Estrutural
    visualizar_estrutura_rede(model)

    # 2. Input Artificial - Visualização de Ativações Intermediárias
    # Criar um input de exemplo aleatório (imagens de 28x28, escala de cinza)
    sample_input = np.random.random((1, 28, 28, 1)).astype('float32')
    visualizar_ativacao_intermediaria(model, sample_input)

    # 3. Input Artificial - Grad-CAM
    visualizar_gradcam(model, sample_input)
