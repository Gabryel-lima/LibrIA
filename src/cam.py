import torch
import cv2
import numpy as np
from pathlib import Path
from conf import Config_Img_Classifier, __DEVICE__, CFG
from GestureNet import ASLNet  # Certifique-se de importar corretamente o modelo
from grad_cam import GradCAM  # Importa o GradCAM real
#import tensorflow as tf

# Carregar configuração
config = Config_Img_Classifier()

def open_camera(ip_url="http://192.168.1.3:4747/video", fallback_device=0):
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

def preprocess_frame_ASLNet(frame, img_size):
    """Converte o frame para grayscale, redimensiona e normaliza."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (img_size, img_size))
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(normalized_frame).float()
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # [Batch, Channel, H, W]
    return input_tensor, resized_frame

def camASLNet():
    try:
        # Carrega o modelo
        model = ASLNet().to(__DEVICE__)
        model.load_state_dict(torch.load(config.BEST_MODEL, map_location=__DEVICE__))
        model.eval()
        print("[INFO] Modelo carregado com sucesso.")

        # Cria o GradCAM
        target_layer = model.features[6]  # <- Ajuste se quiser uma camada melhor
        cam_generator = GradCAM(model, target_layer)

    except Exception as e:
        print(f"[ERROR] Falha ao carregar o modelo: {e}")
        return

    # Tenta abrir a câmera
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

        # Pré-processamento
        input_tensor, gray_frame = preprocess_frame_ASLNet(frame, 64) # config.IMAGE_SIZE
        input_tensor = input_tensor.to(__DEVICE__)

        # Inferência
        try:
            with torch.no_grad():
                output = torch.softmax(model(input_tensor), dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_name = config.LABELS[predicted_label]
        except Exception as e:
            print(f"[ERROR] Falha na inferência: {e}")
            predicted_name = "Erro"
            continue  # Pula esse frame com erro

        # Gera o Grad-CAM verdadeiro
        cam_map = cam_generator.generate_cam(input_tensor)
        heatmap = np.uint8(255 * cam_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # Combina o frame original com o heatmap
        combined = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Overlay de texto no combinado
        cv2.putText(combined, f"Pred: {predicted_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar a imagem combinada
        cv2.imshow(window_name, combined)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC para sair
            print("[INFO] Encerrando...")
            break

    cap.release()
    cv2.destroyAllWindows()
    
#####################################################################################################

import cv2
import numpy as np
import tensorflow as tf
from keras.api.models import load_model, Model
from keras.api.applications.vgg16 import preprocess_input
from conf import CFG

MODEL_PATH = "src/saved/asl_vgg16.keras" # caso eu re-treine o modelo, preciso mudar aqui; asl_vgg16_full_model.keras ou asl_vgg16.keras

def open_camera(device_index=0):
    cap = cv2.VideoCapture(device_index)
    return cap if cap.isOpened() else None

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

    