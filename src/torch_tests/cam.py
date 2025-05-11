import torch
import cv2
import numpy as np
from src.torch_tests.conf import CFG_HybridASLNet, __DEVICE__
from torchvision import transforms

# Importar o modelo e a classe GradCAM
from src.torch_tests.HybridASLNet import HybridASLNet, GradCAM
from src.torch_tests.ASLnet import ASLDataset, ASLNet
from src.torch_tests.conf import CFG_ASLNet

# Normalização da ImageNet
# HybridASLNet usa VGG16 como backbone
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CFG_HybridASLNet.IMG_SIZE, CFG_HybridASLNet.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Função para abrir a câmera
def open_camera(ip_url="/dev/video1", fallback_device=0):
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

def preprocess_frame_hybrid(frame):
    # Converte o frame para RGB, redimensiona e normaliza
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb_frame).unsqueeze(0)  # [1, 3, H, W]
    return tensor.to(__DEVICE__), rgb_frame

def camHybridASLNet():
    # Carrega o modelo HybridASLNet
    try:
        model = HybridASLNet(num_classes=CFG_HybridASLNet.NUM_CLASSES).to(__DEVICE__)
        model.load_state_dict(torch.load(CFG_HybridASLNet.BEST_MODEL, map_location=__DEVICE__))
        model.eval()
        print("[INFO] Modelo HybridASLNet carregado.")

        # Última camada convolucional da VGG16: features[29] (ReLU após conv5_3)
        target_layer = model.features[29]
        cam_generator = GradCAM(model, target_layer)

    except Exception as e:
        print(f"[ERROR] Falha ao carregar o modelo: {e}")
        return

    cap = open_camera()
    if cap is None:
        return

    window_name = "Grad-CAM HybridASLNet"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame inválido recebido. Pulando...")
            continue

        input_tensor, display_frame = preprocess_frame_hybrid(frame)

        try:
            with torch.no_grad():
                output = torch.softmax(model(input_tensor), dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_name = CFG_HybridASLNet.LABELS[predicted_label]
        except Exception as e:
            print(f"[ERROR] Erro de inferência: {e}")
            continue

        cam_map = cam_generator.generate_cam(input_tensor, class_idx=predicted_label)
        heatmap = np.uint8(255 * cam_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        combined = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        cv2.putText(combined, f"Pred: {predicted_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, combined)

        if cv2.waitKey(10) & 0xFF == 27:
            print("[INFO] Encerrando câmera...")
            break

    cap.release()
    cv2.destroyAllWindows()

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
        model.load_state_dict(torch.load(CFG_ASLNet.BEST_MODEL, map_location=__DEVICE__))
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
        input_tensor, gray_frame = preprocess_frame_ASLNet(frame, CFG_ASLNet.IMG_SIZE) # CFG_ASLNet.IMAGE_SIZE
        input_tensor = input_tensor.to(__DEVICE__)

        # Inferência
        try:
            with torch.no_grad():
                output = torch.softmax(model(input_tensor), dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_name = CFG_ASLNet.LABELS[predicted_label]
        except Exception as e:
            print(f"[ERROR] Falha na inferência: {e}")
            predicted_name = "Erro"
            continue  # Pula esse frame com erro

        # Gera o Grad-CAM verdadeiro
        cam_map = cam_generator.generate_cam(input_tensor, class_idx=predicted_label)
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

if __name__ == "__main__":
    #camHybridASLNet()
    camASLNet()
