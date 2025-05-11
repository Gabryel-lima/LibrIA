import cv2
import os
import random
import string
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import tensorflow as tf
import mediapipe as mp
import threading
import time
from collections import Counter
from torchvision.models import mobilenet_v3_small

class CFG:
    TRAIN_PATH = "ASL_Alphabet_Dataset/asl_alphabet_train"
    LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]
    NUM_CLASSES = len(LABELS)
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 25
    LR = 1e-4
    MOMENTUM = 0.9
    SEED = 42
    LANDMARKS_DIM = 21 * 3  # 21 pontos com x, y, z cada

    @staticmethod
    def seed_everything():
        random.seed(CFG.SEED)
        os.environ["PYTHONHASHSEED"] = str(CFG.SEED)
        np.random.seed(CFG.SEED)
        tf.random.set_seed(CFG.SEED)
        torch.manual_seed(CFG.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CFG.SEED)

def _plot_confusion_matrix(preds, labels, plot_dir):
    """Plot confusion matrix at end of training"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=CFG.LABELS,
                yticklabels=CFG.LABELS)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(plot_dir/"confusion_matrix.png")
    plt.close()

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
    # garante gradientes no input
    input_tensor = input_tensor.clone().detach().to(next(model.parameters()).device)
    input_tensor.requires_grad_(True)

    if landmarks_tensor is not None:
        landmarks_tensor = landmarks_tensor.clone().detach().to(next(model.parameters()).device)
        landmarks_tensor.requires_grad_(True)

    activations = []
    gradients   = []

    def forward_hook(module, inp, out):
        # clona e remove do grafo para não ficar uma view
        activations.append(out.clone().detach())
    def backward_hook(module, grad_in, grad_out):
        # clona e remove do grafo para não ficar uma view
        gradients.append(grad_out[0].clone().detach())

    # registra hooks na camada alvo
    if target_layer is None:
        raise ValueError("target_layer must be provided for Grad-CAM")
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    # use full backward hook para versões novas do PyTorch
    try:
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    except AttributeError:
        handle_bwd = target_layer.register_backward_hook(backward_hook)

    # forward
    model.zero_grad()
    output = model(input_tensor, landmarks_tensor) if landmarks_tensor is not None else model(input_tensor)
    if target_class is None:
        target_class = output.softmax(dim=1).argmax(dim=1).item()
    # perda escalar para classe alvo
    loss = output[0, target_class]
    loss.backward()

    # obtém ativação e gradiente
    act = activations[0].detach()    # [1, C, h, w]
    grad= gradients[0].detach()      # [1, C, h, w]

    # peso por canal (global average pooling)
    weights = grad.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
    cam = (weights * act).sum(dim=1).squeeze()     # [h, w]
    cam = torch.relu(cam)
    # normalização
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    return cam

class ASLNetMobile(nn.Module):
    def __init__(self, num_classes=29, temporal_window=5):
        super().__init__()
        self.temporal_window = temporal_window
        
        # Backbone MobileNetV3
        self.backbone = mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove a última camada
        
        # Branch de Landmarks (63 features)
        self.landmarks_fc = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusão Temporal (Média Móvel)
        self.fc = nn.Linear(576 + 64, num_classes)  # 576 features da MobileNet
        
    def forward(self, x, landmarks, history=None):
        # x: [batch_size, 3, 224, 224]
        features = self.backbone(x)  # [batch_size, 576]
        
        # Processa landmarks
        landmarks_feat = self.landmarks_fc(landmarks)  # [batch_size, 64]
        
        # Concatena features
        combined = torch.cat([features, landmarks_feat], dim=1)  # [batch_size, 640]
        
        # Fusão temporal (se history fornecido)
        if history is not None:
            combined = torch.stack(history + [combined], dim=1)  # [batch_size, window, 640]
            combined = torch.mean(combined, dim=1)  # Média móvel
        
        return self.fc(combined)

def extract_hand_landmarks_features(results):
    """
    Extrai características dos landmarks de mãos detectados pelo MediaPipe.
    Retorna um tensor com as coordenadas normalizadas.
    """
    # Inicializa um vetor de características vazio
    landmark_features = []

    if results.multi_hand_landmarks:
        # Pega o primeiro conjunto de landmarks (primeira mão detectada)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extrai coordenadas x, y, z de cada ponto
        for landmark in hand_landmarks.landmark:
            landmark_features.extend([landmark.x, landmark.y, landmark.z])

        # Normaliza as features (opcional mas recomendado)
        if landmark_features:
            landmark_features = np.array(landmark_features)
            min_val = landmark_features.min()
            max_val = landmark_features.max()
            landmark_features = (landmark_features - min_val) / (max_val - min_val + 1e-8)
            landmark_features = landmark_features.tolist()

    # Se não detectou landmarks, retorna vetor de zeros
    if not landmark_features:
        # MediaPipe Hands tem 21 pontos com x,y,z (63 features)
        landmark_features = [0.0] * CFG.LANDMARKS_DIM

    return torch.tensor(landmark_features, dtype=torch.float32)

class LibrasDataset(Dataset):
    def __init__(self, split='train', transform=None, val_ratio=0.2):
        super().__init__()
        self.transform = transform
        samples = []
        for idx, label in enumerate(CFG.LABELS):
            label_dir = os.path.join(CFG.TRAIN_PATH, label)
            if not os.path.isdir(label_dir): continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    samples.append((os.path.join(label_dir, fname), idx))
        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - val_ratio))
        self.data = samples[:split_idx] if split == 'train' else samples[split_idx:]
        print(f"{split}: {len(self.data)} samples loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Training and evaluation
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_accum = 0.0
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_accum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return loss_accum / len(loader), correct / total if total > 0 else 0.0

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_accum = 0.0
    correct = total = 0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_accum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Coleta predições e labels para matriz de confusão
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # Concatena todos os batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return (loss_accum / len(loader),
            correct / total if total > 0 else 0.0,
            all_preds,
            all_labels)

# Main script
def main():
    CFG.seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_ds = LibrasDataset('train', transform)
    val_ds   = LibrasDataset('val',   transform)
    train_dl = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)

    model = ASLNetMobile(feature_dim=512).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CFG.LR, momentum=CFG.MOMENTUM)

    from pathlib import Path
    plot_dir = Path("./plots")
    plot_dir.mkdir(exist_ok=True)

    best_val = float('inf')
    for epoch in range(1, CFG.EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion, device)
        vl_loss, vl_acc, vl_preds, vl_labels = eval_epoch(model, val_dl, criterion, device)
        print(f"[Epoch {epoch:02d}/{CFG.EPOCHS}] "
              f"Train L: {tr_loss:.4f}, A: {tr_acc:.4%} | "
              f"Val L: {vl_loss:.4f}, A: {vl_acc:.4%}")

        # gera e salva a matriz a cada época
        _plot_confusion_matrix(vl_preds, vl_labels, plot_dir)

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), "filter_static_landmarks_best.pt")
            print("👉 Best landmarks model saved")

    torch.save(model.state_dict(), "filter_static_landmarks_final.pt")
    print("👉 Landmarks final model saved")

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


def inference_with_gradcam_and_hands(model_path, source=0, target_layer_name='conv5_3', inference_fps=10):
    # ------------------------
    # 1) Configurações iniciais
    # ------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLNetMobile().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # ------------------------------------------------
    # 2) Inicia a captura em background com VideoStream
    # ------------------------------------------------
    vs = VideoStream(source=source)
    # calcula o intervalo entre quadros para manter FPS constante
    frame_interval = 1.0 / inference_fps

    # --------------------------------------------
    # 3) Inicializa o MediaPipe Hands e o desenho
    # --------------------------------------------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    drawing_spec = mp_draw.DrawingSpec(
        thickness=2, circle_radius=4, color=(0, 0, 255)
    )

    # ------------------------------------------------
    # 4) Seleciona a camada de interesse para o Grad-CAM
    # ------------------------------------------------
    target_layer = getattr(model, target_layer_name, None)
    if not isinstance(target_layer, nn.Conv2d):
        # se não existir ou não for Conv2d, pega o último conv da lista
        for m in reversed(model.vgg_feats):
            if isinstance(m, nn.Conv2d):
                target_layer = m
                break
    print(f">>> {target_layer} selecionada como target layer")

    # ------------------------
    # 5) Variáveis para estabilização de predição
    # ------------------------
    prediction_history = []
    history_size = 5  # Tamanho da janela de estabilização
    current_prediction = 0  # Predição inicial
    min_confidence = 0.6  # Limite mínimo de confiança

    # ------------------------
    # 6) Loop principal de inferência
    # ------------------------
    try:
        while True:
            loop_start = time.time()

            # a) lê o frame mais novo
            ret, frame = vs.read()
            if not ret:
                break

            # b) converte para RGB e processa mãos
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # c) Extrai landmarks como features
            landmark_features = extract_hand_landmarks_features(results)
            landmark_features = landmark_features.unsqueeze(0).to(device)  # Adiciona dim de batch

            # d) desenha os landmarks detectados
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_spec, drawing_spec
                    )

            # e) Prepara a imagem para o modelo
            inp = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

            # f) gera o Grad-CAM com landmarks
            cam = apply_gradcam_torch(model, inp, landmark_features, None, target_layer)
            heatmap = cv2.resize(
                (cam * 255).astype('uint8'),
                (frame.shape[1], frame.shape[0])
            )
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            # g) faz a predição com landmarks e obtém o rótulo
            with torch.no_grad():
                output = model(inp, landmark_features)
                confidence = torch.nn.functional.softmax(output, dim=1).max().item()
                pred = output.softmax(1).argmax(1).item()

                # Só atualiza se a confiança for alta o suficiente
                if confidence > min_confidence:
                    # Adiciona à história para estabilização
                    prediction_history.append(pred)
                    if len(prediction_history) > history_size:
                        prediction_history.pop(0)

                    # Usa a predição mais frequente
                    most_common_pred = Counter(prediction_history).most_common(1)[0][0]
                    current_prediction = most_common_pred

            # Usa a predição estabilizada
            label = CFG.LABELS[current_prediction]

            # h) Mostra informações na tela
            cv2.putText(
                overlay, f"Pred: {label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2
            )

            # i) exibe o resultado na janela
            cv2.imshow('Hands + Grad-CAM + Landmarks', overlay)

            # j) throttle para manter FPS constante
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            # k) condicional de saída ('q' para sair)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # ----------------------------
        # 7) Limpeza de recursos
        # ----------------------------
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # inference_with_gradcam_and_hands(
    #     "src/model/filters/v1/v1_static_91.pt",  # Atualize para o caminho do seu modelo
    #     source=0,  # Use 0 para webcam padrão
    #     target_layer_name='conv5_3',
    #     inference_fps=15  # Ajuste conforme FPS desejado e capacidade do hardware
    # )
    main()
