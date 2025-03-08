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

def cam():
    try:
        # Carrega o modelo corretamente
        model = ASLNet().to(__DEVICE__)
        model.load_state_dict(torch.load(config.BEST_MODEL, map_location=__DEVICE__))
        model.eval()
    except Exception as e:
        print("Erro ao carregar o modelo:", e)
        return

    # Abre a câmera com fallback
    cap = cv2.VideoCapture("http://192.168.1.3:4747/video")
    if not cap.isOpened():
        print("Erro ao abrir a câmera IP. Tentando câmera local...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Nenhuma câmera disponível!")
            return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break

        # Converte para RGB e redimensiona
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (config.IMG_SIZE, config.IMG_SIZE))
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        # Converte para tensor do PyTorch
        input_tensor = torch.from_numpy(normalized_frame).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(__DEVICE__)

        # Inferência
        try:
            with torch.no_grad():
                output = torch.softmax(model(input_tensor), dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            print(f"Inferência realizada com sucesso! Classe prevista: {config.LABELS[predicted_label]}")
        except Exception as e:
            print("Erro na inferência:", e)

        # Exibir câmera
        cv2.imshow("Câmera + Inferência", frame)
        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam()
