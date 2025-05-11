import tensorflow as tf
from keras.api import layers, Model, Input
from keras.api.models import load_model, Sequential
import cv2
import numpy as np

# Constantes
IMG_SIZE    = 400
NUM_CLASSES = 26
LABELS      = [chr(i) for i in range(65, 65+26)]  # A-Z
MODEL_PATH = 'src/model/filters/tf_classifier_cnn.keras'

class CNNClassifier(Model):
    """https://www.kaggle.com/code/gabryel27/cnn-model-for-asl"""
    def __init__(self, img_size: int, num_classes: int):
        super(CNNClassifier, self).__init__()
        
        # Primeiro bloco convolucional
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', 
                                   input_shape=(img_size, img_size, 3))
        self.bn1   = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.drop1 = layers.Dropout(0.25)
        
        # Segundo bloco convolucional
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.bn2   = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.drop2 = layers.Dropout(0.25)
        
        # Terceiro bloco convolucional
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.bn3   = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.drop3 = layers.Dropout(0.25)
        
        # Quarto bloco convolucional
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu')
        self.bn4   = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D((2, 2))
        self.drop4 = layers.Dropout(0.25)
        
        # Camadas finais (flatten + densas)
        self.flatten = layers.Flatten()
        
        self.dense1 = layers.Dense(512, activation='relu')
        self.bn_dense = layers.BatchNormalization()
        self.drop_dense = layers.Dropout(0.5)
        
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Bloco 1
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        
        # Bloco 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        
        # Bloco 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        x = self.drop3(x, training=training)
        
        # Bloco 4
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.pool4(x)
        x = self.drop4(x, training=training)
        
        # Flatten + Dense
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn_dense(x, training=training)
        x = self.drop_dense(x, training=training)
        
        return self.classifier(x)
    
# # Instanciação e compilação
# inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# model  = CNNClassifier(num_classes=NUM_CLASSES)
# _      = model(inputs)  # dispara o build
# model.compile(optimizer='adam',
#             loss='categorical_crossentropy',
#             metrics=['accuracy'])

# model.summary()

# CNNClassifier(IMG_SIZE, NUM_CLASSES).summary()

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

def preprocess_frame(frame):
    # BGR -> RGB, redimensiona, normaliza [0,1]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    arr     = resized.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0), resized

def make_gradcam(model, img_tensor, layer_name="conv2d_3"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
    heatmap = cv2.resize(cam.numpy(), (IMG_SIZE, IMG_SIZE))
    return heatmap

def overlay_heatmap(frame, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1-alpha, jet, alpha, 0)

def apply_gradcam_overlay(model, inp, frame, target_layer="conv2d_3"):
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

def cam(use_gradcam=True, target_layer="conv2d_3"):
    model = load_model(MODEL_PATH)#, custom_objects={"CNNClassifier": CNNClassifier})
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
    cam(use_gradcam=True)
