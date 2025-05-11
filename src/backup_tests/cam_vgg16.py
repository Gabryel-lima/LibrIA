"""Teste de inferência com Grad-CAM usando VGG16 para ASL (A-Z + 3 especiais)"""
#             continue  # Pula esse frame com erro
import cv2
import numpy as np
import tensorflow as tf
from keras.api.applications import VGG16
from keras.api.models import Model
from keras.api.layers import Dense, Flatten, Dropout
from keras.api.preprocessing.image import img_to_array
from keras.api.applications.vgg16 import preprocess_input
import os

# Lista de rótulos conforme o ASL (A-Z + 3 especiais)
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# Caminho dos pesos treinados
MODEL_WEIGHTS_PATH = "src/saved/asl_vgg16_best_weights.keras"

def open_camera(ip_url="http://192.168.1.3:4747/video", fallback_device=0):
    cap = cv2.VideoCapture(ip_url)
    if cap.isOpened():
        print("[INFO] Câmera IP conectada com sucesso.")
        return cap
    else:
        print("[WARN] Falha ao conectar à câmera IP. Tentando câmera local...")
        cap.release()
        cap = cv2.VideoCapture(fallback_device)
        if cap.isOpened():
            print("[INFO] Câmera local conectada com sucesso.")
            return cap
        else:
            print("[ERROR] Nenhuma câmera disponível!")
            return None

def preprocess_frame_ImageNet(frame, img_size=224):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (img_size, img_size))
    img_array = img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, resized_frame

def create_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(LABELS), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model

def generate_gradcam(model, image_tensor, target_layer_name="block5_conv3"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / tf.math.reduce_max(cam)
    cam = tf.image.resize(cam[..., tf.newaxis], (224, 224)).numpy()
    return cam.squeeze()

def camImageNet():
    try:
        model = create_model()
        print("[INFO] Modelo VGG16 carregado com sucesso.")
    except Exception as e:
        print(f"[ERROR] Falha ao carregar o modelo: {e}")
        return

    cap = open_camera()
    if cap is None:
        return

    window_name = "Câmera + Inferência + Grad-CAM"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Frame inválido recebido. Tentando novamente...")
            continue

        input_tensor, original_frame = preprocess_frame_ImageNet(frame, 224)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        try:
            predictions = model(input_tensor, training=False)
            predicted_label = tf.argmax(predictions[0]).numpy()
            predicted_name = LABELS[predicted_label]
        except Exception as e:
            print(f"[ERROR] Falha na inferência: {e}")
            predicted_name = "Erro"
            continue

        cam_map = generate_gradcam(model, input_tensor)
        heatmap = np.uint8(255 * cam_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        combined = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        cv2.putText(combined, f"Pred: {predicted_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, combined)

        if cv2.waitKey(10) & 0xFF == 27:
            print("[INFO] Encerrando...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camImageNet()
