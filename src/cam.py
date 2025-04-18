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

"""Api keras legacy"""
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras._tf_keras.keras.applications.vgg16 import VGG16, preprocess_input
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint

# Configuração
MODEL_WEIGHTS_PATH = "src/saved/asl_vgg16_best_weights.keras"

def preprocess_frame_ImageNet(frame, img_size=224):
    """ Converte o frame para RGB, redimensiona e normaliza para a ImageNet """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte para RGB
    resized_frame = cv2.resize(frame_rgb, (img_size, img_size))  # Redimensiona para 224x224
    img_array = img_to_array(resized_frame)  # Converte para array
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona batch dimension
    img_array = preprocess_input(img_array)  # Pré-processamento específico para ImageNet
    return img_array, resized_frame

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

# Configuração de modelo
def create_model():
    """ Cria o modelo VGG16 com camadas congeladas e novas camadas para classificação """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layer in base_model.layers:
        layer.trainable = False  # Congelar as camadas do modelo pré-treinado

    # Adicionar camadas customizadas para classificação
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(29, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(MODEL_WEIGHTS_PATH)  # Carregar pesos salvos
    model.evaluate()  # Colocar o modelo em modo de avaliação
    model.summary()
    return model

def camImageNet():
    """ Aplica Grad-CAM usando o modelo VGG16 """
    try:
        model = create_model()  # Cria o modelo com o VGG16

        # Defina a camada alvo para o Grad-CAM (última camada convolucional de VGG16)
        target_layer = model.get_layer("block5_conv3")
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

        # Pré-processamento do frame
        input_tensor, gray_frame = preprocess_frame_ImageNet(frame, 224)  # Para o VGG16
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        # Inferência
        try:
            output = model(input_tensor, training=False)
            predicted_label = tf.argmax(output[0]).numpy()
            predicted_name = CFG.labels[predicted_label]
        except Exception as e:
            print(f"[ERROR] Falha na inferência: {e}")
            predicted_name = "Erro"
            continue  # Pula este frame em caso de erro

        # Gera o Grad-CAM
        cam_map = generate_gradcam(model, input_tensor)
        heatmap = np.uint8(255 * cam_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        combined = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Combina a imagem original com o mapa de calor
        combined = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Overlay de texto com a previsão na imagem combinada
        cv2.putText(combined, f"Pred: {predicted_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar a imagem combinada
        cv2.imshow(window_name, combined)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC para sair
            print("[INFO] Encerrando...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camASLNet()
    #camImageNet()
    