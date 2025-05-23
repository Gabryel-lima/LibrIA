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
from pathlib import Path
from tqdm import tqdm

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
    
def _plot_calibration_curve(preds, labels, confs, plot_dir, n_bins=10):
    """
    Plota a curva de calibração (Reliability Diagram) comparando
    confiança média em cada bin com a acurácia real daquele bin.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # bins uniformes de 0.0 a 1.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    accuracies = []
    confidences = []

    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs >= start) & (confs < end)
        if mask.sum() > 0:
            accuracies.append((preds[mask] == labels[mask]).mean())
            confidences.append(confs[mask].mean())
        else:
            accuracies.append(np.nan)
            confidences.append((start + end) / 2.0)

    plt.figure(figsize=(6, 6))
    plt.plot(confidences, accuracies, marker='o', label='Empirical')
    plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
    plt.title("Reliability Diagram")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(str(plot_dir / "calibration_curve.png"))
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

class ASLNetVGG(nn.Module):
    def __init__(self, feature_dim=512, landmarks_dim=CFG.LANDMARKS_DIM, freeze_vgg=True):
        super().__init__()
        # Backbone VGG16 pretrained
        vgg = models.vgg16(pretrained=True)
        # 1) DESATIVA todo ReLU(inplace=True) → ReLU(inplace=False)
        for m in vgg.features.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        # Convolutional features
        self.vgg_feats = vgg.features
        if freeze_vgg:
            for p in self.vgg_feats.parameters():
                p.requires_grad = False
        # Pooling for static feature (1x1)
        self.pool1 = nn.AdaptiveAvgPool2d((1,1))
        # Pooling for classifier branch (7x7)
        self.pool2 = vgg.avgpool
        # Classifier branch (penultimate layers)
        orig_cls = list(vgg.classifier.children())[:-1]
        self.cls_feats = nn.Sequential(*orig_cls)

        # Nova branch para landmarks
        self.lm_fc = nn.Sequential(
            nn.Linear(landmarks_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

        # Projection (agora com landmarks)
        self.proj = nn.Linear(512 + 4096 + 128, feature_dim)
        self.act  = nn.ReLU()
        # Final head
        self.classifier = nn.Linear(feature_dim, CFG.NUM_CLASSES)

    def forward(self, x, landmarks=None):
        # 1) extrai features de imagem
        feats = self.vgg_feats(x)           # [B,512,7,7]
        f1   = self.pool1(feats).flatten(1) # [B,512]
        f2   = self.cls_feats(self.pool2(feats).flatten(1))  # [B,4096]

        # 2) prepara landmarks (zeros se None)
        if landmarks is None:
            batch = x.size(0)
            landmarks = torch.zeros(batch, CFG.LANDMARKS_DIM, device=x.device)
        f3 = self.lm_fc(landmarks)   # [B,128]

        # 3) concatena sempre os três vetores
        f  = torch.cat([f1, f2, f3], dim=1) # [B,512+4096+128 = 4736]
        feat = self.act(self.proj(f))       # [B, feature_dim]
        return self.classifier(feat)        # [B, NUM_CLASSES]

    @property
    def conv5_3(self):
        # retorna o último Conv2d sem registrá-lo no state_dict
        return self.vgg_feats[28]

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

# ------------------------------------------------
# Treino e validação
# ------------------------------------------------
def run_epoch(model, train_dl, val_dl, opt, crit, scaler, epoch):
    # total batches: treino + validação
    total_batches = len(train_dl) + len(val_dl)
    pbar = tqdm(total=total_batches,
                desc=f"Epoch {epoch}/{CFG.EPOCHS}",
                unit="batch")

    # --- Treino ---
    model.train()
    running_loss = running_corr = running_tot = 0
    for imgs, lms, lbls in train_dl:
        imgs, lms, lbls = imgs.to(CFG.DEVICE), lms.to(CFG.DEVICE), lbls.to(CFG.DEVICE)
        opt.zero_grad()
        with torch.autocast(enabled=CFG.USE_AMP):
            out = model(imgs, lms)
            loss = crit(out, lbls)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        running_loss += loss.item() * lbls.size(0)
        preds = out.argmax(1)
        running_corr += (preds == lbls).sum().item()
        running_tot += lbls.size(0)

        # Atualiza a barra
        avg_loss = running_loss / running_tot
        acc = running_corr / running_tot
        pbar.set_postfix(train_loss=f"{avg_loss:.4f}", train_acc=f"{acc:.2%}")
        pbar.update(1)

    # --- Validação ---
    model.eval()
    val_loss = val_corr = val_tot = 0
    all_preds, all_labels, all_confs = [], [], []
    with torch.no_grad():
        for imgs, lms, lbls in val_dl:
            imgs, lms, lbls = imgs.to(CFG.DEVICE), lms.to(CFG.DEVICE), lbls.to(CFG.DEVICE)
            out = model(imgs, lms)
            loss = crit(out, lbls)

            val_loss += loss.item() * lbls.size(0)
            probs = torch.softmax(out, dim=1)
            confs, preds = probs.max(dim=1)
            val_corr += (preds == lbls).sum().item()
            val_tot += lbls.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(lbls.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(lbls.cpu().numpy())
            all_confs.append(confs.cpu().numpy())

            avg_vloss = val_loss / val_tot
            vacc = val_corr / val_tot
            pbar.set_postfix(val_loss=f"{avg_vloss:.4f}", val_acc=f"{vacc:.2%}")
            pbar.update(1)

    pbar.close()
    return (running_loss / running_tot, running_corr / running_tot,
            val_loss / val_tot, val_corr / val_tot,
            np.concatenate(all_preds), 
            np.concatenate(all_labels), 
            np.concatenate(all_confs))

# Main script
def main():
    CFG.seed_everything()
    
    tfm = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE,CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    train_ds = LibrasDataset('train', tfm)
    val_ds   = LibrasDataset('val', tfm)
    train_dl = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model = ASLNetVGG().to(CFG.DEVICE)
    
    crit  = nn.CrossEntropyLoss()
    opt   = optim.SGD(model.parameters(), lr=CFG.LR, momentum=CFG.MOMENTUM)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)
    scl   = torch.GradScaler()
    
    pdir = Path('./plots'); pdir.mkdir(exist_ok=True)
    
    best = float('inf')
    for epoch in range(1, CFG.EPOCHS + 1):
        tr_l, tr_a, vl_l, vl_a, vp, vl, v_conf = run_epoch(
            model, train_dl, val_dl, opt, crit, scl, epoch
        )
        
        sch.step(vl_l)
        _plot_confusion_matrix(vp, vl, pdir)
        _plot_calibration_curve(vp, vl, v_conf, pdir)
        
        if vl_l < best:
            best = vl_l
            torch.save(model.state_dict(), 'best_landmarks.pt')

    torch.save(model.state_dict(), 'final_landmarks.pt')

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
    model = ASLNetVGG().to(device)
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
        max_num_hands=1,
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
            label_idx_pred = CFG.LABELS[pred]
            
            # h) Mostra informações na tela
            cv2.putText(
                overlay, f"Pred: {label_idx_pred} ({confidence:.2f})", (10, 30),
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
        hands.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_with_gradcam_and_hands(
        "src/model/filters/v3/v3_land_63.pt",
        source=0,  # Use 0 para webcam padrão
        target_layer_name='conv5_3',
        inference_fps=15  # Ajuste conforme FPS desejado e capacidade do hardware
    )
