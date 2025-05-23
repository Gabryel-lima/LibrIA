# 1. Importações e Configurações
import os, random, string
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import mediapipe as mp
import threading
import time
from collections import Counter
from PIL import Image

class CFG:
    TRAIN_PATH   = "/kaggle/input/aslamerican-sign-language-aplhabet-dataset/ASL_Alphabet_Dataset/asl_alphabet_train"
    IMG_SIZE     = 224
    BATCH_SIZE   = 64
    EPOCHS       = 20
    LR           = 3e-4
    WEIGHT_DECAY = 1e-4
    VAL_RATIO    = 0.2
    SEQ_LEN      = 5
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LABELS       = list(string.ascii_uppercase) + ["del", "nothing", "space"]

class ASLNet(nn.Module):
    def __init__(self, num_classes=len(CFG.LABELS)):
        super().__init__()
        self.backbone = mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(576, num_classes)

    def forward(self, x, landmarks=None):
        feat = self.backbone(x)
        return self.fc(feat)

model = ASLNet().to(CFG.DEVICE)

# --------------------------------------------------
# Classe que faz a captura contínua em thread separada
# --------------------------------------------------
class VideoStream:
    def __init__(self, source=0):
        # abre o device de vídeo (webcam ou DroidCam)
        self.cap = cv2.VideoCapture(source)
        # lê o primeiro frame
        self.grabbed, self.frame = self.cap.read()
        # flag para encerrar a thread
        self.stopped = False
        # inicia a thread de captura em background
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        # loop de captura: sempre atualiza self.frame com o último frame disponível
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()

    def read(self):
        # retorna o frame mais recente lido pela thread
        return self.grabbed, self.frame

    def stop(self):
        # sinaliza para encerrar o loop e libera o dispositivo
        self.stopped = True
        self.cap.release()

def apply_gradcam_torch(model, input_tensor, landmarks_tensor=None, target_class=None, target_layer=None):
    """
    Gera mapa de calor Grad-CAM para um modelo PyTorch.
    Args:
        model: instância do modelo PyTorch.
        input_tensor: Tensor[1,C,H,W] pré-processado.
        landmarks_tensor: Tensor de landmarks (opcional).
        target_class: índice da classe-alvo; se None, usa predição.
        target_layer: camada convolucional alvo (nn.Module) do modelo.
    Retorna:
        cam: np.ndarray 2D normalizado.
    """
    input_tensor = input_tensor.clone().detach().to(next(model.parameters()).device)
    input_tensor.requires_grad_(True)

    if landmarks_tensor is not None:
        landmarks_tensor = landmarks_tensor.clone().detach().to(next(model.parameters()).device)
        landmarks_tensor.requires_grad_(True)

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.clone().detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].clone().detach())

    if target_layer is None:
        raise ValueError("target_layer must be provided for Grad-CAM")
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    try:
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    except AttributeError:
        handle_bwd = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor, landmarks_tensor) if landmarks_tensor is not None else model(input_tensor)
    if target_class is None:
        target_class = output.softmax(dim=1).argmax(dim=1).item()
    loss = output[0, target_class]
    loss.backward()

    act = activations[0].detach()
    grad = gradients[0].detach()

    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    handle_fwd.remove()
    handle_bwd.remove()

    return cam

def extract_hand_landmarks_features(results):
    """
    Extrai características dos landmarks de mãos detectados pelo MediaPipe.
    Retorna um tensor com as coordenadas normalizadas.
    """
    landmark_features = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            landmark_features.extend([lm.x, lm.y, lm.z])
        arr = np.array(landmark_features)
        mn = arr.min()
        mx = arr.max()
        landmark_features = ((arr - mn) / (mx - mn + 1e-8)).tolist()

    if not landmark_features:
        landmark_features = [0.0] * (21 * 3)

    return torch.tensor(landmark_features, dtype=torch.float32)

def inference_with_gradcam_and_hands(model_path, source=0, target_layer_name=None, inference_fps=10):
    device = CFG.DEVICE
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    vs = VideoStream(source)
    frame_interval = 1.0 / inference_fps

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    drawing_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=4, color=(0, 0, 255))

    # Seleciona camada para Grad-CAM
    if target_layer_name and hasattr(model, target_layer_name):
        target_layer = getattr(model, target_layer_name)
    else:
        conv_layers = [m for m in model.backbone.features if isinstance(m, nn.Conv2d)]
        target_layer = conv_layers[-1] if conv_layers else None
    print(f">>> Target layer selecionada para Grad-CAM: {target_layer}")

    prediction_history = []
    history_size = CFG.SEQ_LEN
    current_prediction = 0
    min_confidence = 0.6

    try:
        while True:
            loop_start = time.time()
            ret, frame = vs.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            landmark_features = extract_hand_landmarks_features(results)
            landmark_features = landmark_features.unsqueeze(0).to(device)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_spec, drawing_spec
                    )

            inp = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

            cam = apply_gradcam_torch(model, inp, landmark_features, None, target_layer)
            heatmap = cv2.resize((cam * 255).astype('uint8'), (frame.shape[1], frame.shape[0]))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            with torch.no_grad():
                output = model(inp, landmark_features)
                probs = torch.softmax(output, dim=1)
                confidence, pred = probs.max(dim=1).item(), probs.argmax(dim=1).item()
                if confidence > min_confidence:
                    prediction_history.append(pred)
                    if len(prediction_history) > history_size:
                        prediction_history.pop(0)
                    current_prediction = Counter(prediction_history).most_common(1)[0][0]
                else:
                    pred = current_prediction
                    confidence = 0.0

            label = CFG.LABELS[current_prediction]
            cv2.putText(
                overlay, f"Pred: {label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

            cv2.imshow('Hands + Grad-CAM + Landmarks', overlay)

            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        vs.stop()
        hands.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_with_gradcam_and_hands(model_path="")
