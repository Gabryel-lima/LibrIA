import cv2
import numpy as np
import tensorflow as tf
from keras.api.models import load_model, Model
from keras.api.applications.vgg16 import preprocess_input


def open_camera(ip_url="/dev/video1", fallback_device=0):
    # Lembrando, inicie o droidcam em um terminal anterior, antes de passar o código.
    """Tenta abrir a câmera IP; se falhar, tenta abrir a câmera local."""
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

import string

# Labels (29 classes)
LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]
MODEL_PATH = "src/saved/asl_vgg16.keras" # caso eu re-treine o modelo, preciso mudar aqui; asl_vgg16_full_model_chpt.keras ou asl_vgg16_full_model.keras

def preprocess_frame(frame):
    # Redimensiona para o tamanho com que o modelo foi treinado
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (64, 64))
    arr     = np.expand_dims(resized.astype("float32"), axis=0)
    return preprocess_input(arr), resized

def generate_gradcam(model, image_tensor, frame_shape, target_layer_name="block5_conv3"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image_tensor)
        idx = tf.argmax(preds[0])
        loss = preds[:, idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv = conv_outputs[0]
    cam = tf.reduce_sum(pooled * conv, axis=-1).numpy()
    
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (frame_shape[1], frame_shape[0]))  # width x height
    return cam

def apply_gradcam_overlay(model, inp, frame, target_layer="block5_conv3"):
    """
    Gera o mapa de calor Grad-CAM e sobrepõe na imagem original.
    
    Args:
        model (keras.Model): Modelo treinado.
        inp (np.ndarray): Entrada pré-processada para o modelo (1, H, W, C).
        frame (np.ndarray): Frame original da câmera (BGR).
        target_layer (str): Nome da camada convolucional alvo.

    Returns:
        np.ndarray: Frame com sobreposição do mapa de calor.
    """
    # Grad-CAM
    grad_model = Model(inputs=model.inputs, 
                        outputs=[model.get_layer(target_layer).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inp)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv = conv_outputs[0]
    cam = tf.reduce_sum(pooled_grads * conv, axis=-1).numpy()

    # Normalização e redimensionamento
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))  # width x height
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Combinação com o frame original
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay, int(class_idx)

def cam_imagenet(use_gradcam=True, target_layer="block5_conv3"):
    model = load_model(MODEL_PATH)
    cap = open_camera()

    if cap is None:
        print("[ERROR] Não foi possível abrir a câmera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        inp, _ = preprocess_frame(frame)
        preds = model.predict(inp, verbose=0)
        class_idx = int(tf.argmax(preds[0]))
        label = LABELS[class_idx]

        if use_gradcam:
            overlay, _ = apply_gradcam_overlay(model, inp, frame, target_layer=target_layer)
        else:
            overlay = frame.copy()

        cv2.putText(overlay, f"Pred: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Grad-CAM" if use_gradcam else "Prediction", overlay)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_imagenet(use_gradcam=True)
