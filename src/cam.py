import torch
import cv2
import numpy as np
from pathlib import Path
from conf import Config_Img_Classifier
from GestureNet import ASLNet  # Certifique-se de importar corretamente o modelo

# Configuração do dispositivo
__DEVICE__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def preprocess_frame(frame, img_size):
    """Converte o frame para grayscale, redimensiona e normaliza."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # <= AQUI!
    resized_frame = cv2.resize(gray_frame, (img_size, img_size))
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(normalized_frame).float()
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # [Batch, Channel, H, W]
    return input_tensor

def cam():
    try:
        # Carrega o modelo
        model = ASLNet().to(__DEVICE__)
        model.load_state_dict(torch.load(config.BEST_MODEL, map_location=__DEVICE__))
        model.eval()
        print("[INFO] Modelo carregado com sucesso.")
    except Exception as e:
        print(f"[ERROR] Falha ao carregar o modelo: {e}")
        return

    # Tenta abrir a câmera
    cap = open_camera()
    if cap is None:
        return

    window_name = "Câmera + Inferência"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Cria apenas UMA janela

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[WARN] Frame inválido recebido. Tentando novamente...")
            continue  # Não tenta exibir frame quebrado!

        # Pré-processamento
        input_tensor = preprocess_frame(frame, config.IMG_SIZE).to(__DEVICE__)

        # Inferência
        try:
            with torch.no_grad():
                output = torch.softmax(model(input_tensor), dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_name = config.LABELS[predicted_label]
        except Exception as e:
            print(f"[ERROR] Falha na inferência: {e}")
            predicted_name = "Erro"

        # Overlay de texto na imagem
        cv2.putText(frame, f"Pred: {predicted_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar a imagem sempre na mesma janela
        cv2.imshow(window_name, frame)

        # Esperar pela tecla ESC (27) para sair
        if cv2.waitKey(10) & 0xFF == 27:
            print("[INFO] Encerrando...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam()
