import cv2
import numpy as np
import tensorflow as tf
from keras.api.models import load_model, Model
from keras.api.applications.vgg16 import preprocess_input
from conf import CFG

def open_camera(ip_url="/dev/video1", fallback_device=0):
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

MODEL_PATH = "src/saved/asl_vgg16.keras" # caso eu re-treine o modelo, preciso mudar aqui; asl_vgg16_full_model_chpt.keras ou asl_vgg16_full_model.keras

def preprocess_frame(frame):
    # Redimensiona para o tamanho com que o modelo foi treinado
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (CFG.img_width, CFG.img_height))
    arr     = np.expand_dims(resized.astype("float32"), axis=0)
    return preprocess_input(arr), resized

def generate_gradcam(model, image_tensor, target_layer_name="block5_conv3"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image_tensor)
        idx  = tf.argmax(preds[0])
        loss = preds[:, idx]
    grads  = tape.gradient(loss, conv_outputs)[0]
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv   = conv_outputs[0]
    cam    = tf.reduce_sum(pooled * conv, axis=-1)
    cam    = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    cam    = tf.image.resize(cam[..., tf.newaxis], image_tensor.shape[1:3]).numpy().squeeze()
    return cam

def cam_imagenet():
    """
    Perform real-time Grad-CAM visualization using a pre-trained model and a webcam feed.

    This function captures video frames from a webcam, processes each frame through a 
    pre-trained model to generate predictions, and overlays a Grad-CAM heatmap on the 
    original frame to highlight regions of interest. The combined visualization is 
    displayed in a window.

    Returns:
        None

    Notes:
        - Ensure that the `MODEL_PATH` variable points to the correct model file.
        - The `CFG.labels` should contain the class labels corresponding to the model's output.
        - The `generate_gradcam` and `preprocess_frame` functions must be implemented 
            to handle Grad-CAM generation and frame preprocessing, respectively.
        - Press the "Esc" key to exit the visualization.

    Dependencies:
        - OpenCV (cv2) for video capture and visualization.
        - NumPy for numerical operations.
        - A deep learning framework (e.g., TensorFlow or PyTorch) for model loading and predictions.

    Error Handling:
        - If the webcam cannot be opened, an error message is printed, and the function exits.
        - If a frame cannot be read from the webcam, the function skips to the next iteration.

    Example:
        cam_imagenet()
    """
    model = load_model(MODEL_PATH)
    cap   = open_camera()
    if cap is None:
        print("[ERROR] Não foi possível abrir a câmera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        inp, _      = preprocess_frame(frame)
        preds       = model.predict(inp, verbose=0)
        label       = CFG.labels[np.argmax(preds[0])]

        cam_map     = generate_gradcam(model, inp)
        heat        = np.uint8(255 * cam_map)
        heat        = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat        = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
        combined    = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)

        cv2.putText(combined, f"Pred: {label}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Grad-CAM", combined)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_imagenet()
