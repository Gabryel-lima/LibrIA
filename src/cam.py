import torch
import cv2
import numpy as np
from conf import Config_Img_Classifier, __DEVICE__
from torchvision import transforms
from ASLnet import HybridASLNet, GradCAM

# Carregar configuração
data_config = Config_Img_Classifier()

# Normalização da ImageNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((data_config.IMG_SIZE, data_config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb_frame).unsqueeze(0)  # [1, 3, H, W]
    return tensor.to(__DEVICE__), rgb_frame

def camHybridASLNet():
    try:
        model = HybridASLNet(num_classes=data_config.NUM_CLASSES).to(__DEVICE__)
        model.load_state_dict(torch.load(data_config.BEST_MODEL, map_location=__DEVICE__))
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
            predicted_name = data_config.LABELS[predicted_label]
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

if __name__ == "__main__":
    camHybridASLNet()
