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

class CFG:
    #TRAIN_PATH = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
    LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]
    NUM_CLASSES = len(LABELS)
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 25
    LR = 1e-4
    MOMENTUM = 0.9
    SEED = 42

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
            
def apply_gradcam_torch(model, input_tensor, target_class=None, target_layer=None):
    """
    Gera mapa de calor Grad-CAM para um modelo PyTorch.
    Args:
        model: instância do modelo PyTorch.
        input_tensor: Tensor[1,C,H,W] pré-processado.
        target_class: índice da classe-alvo; se None, usa predição.
        target_layer: camada convolucional alvo (nn.Module) do modelo.
    Retorna:
        cam: np.ndarray 2D normalizado.
    """
    # garante gradientes no input
    input_tensor = input_tensor.clone().detach().to(next(model.parameters()).device)
    input_tensor.requires_grad_(True)

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
    output = model(input_tensor)
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
    def __init__(self, feature_dim=512, freeze_vgg=True): # feature_dim=512: static_87.pt, static_91.pt, static_md_67.pt test_static_md.pt
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
        self.asl_feats = nn.Sequential(*orig_cls)
        # Projection
        self.proj = nn.Linear(512 + 4096, feature_dim)
        self.act  = nn.ReLU()
        # Final head
        self.classifier = nn.Linear(feature_dim, CFG.NUM_CLASSES)

    def forward(self, x):
        # x: [B, C, H, W]
        feats = self.vgg_feats(x)         # [B,512,7,7]
        # static path
        f1 = self.pool1(feats)           # [B,512,1,1]
        f1 = torch.flatten(f1,1)         # [B,512]
        # classifier path
        f2 = self.pool2(feats)           # [B,512,7,7]
        f2 = torch.flatten(f2,1)         # [B,25088]
        f2 = self.asl_feats(f2)          # [B,4096]
        # concat + proj
        f  = torch.cat([f1, f2], dim=1)  # [B,4608]
        feat = self.act(self.proj(f))     # [B,feature_dim]
        return self.classifier(feat)      # [B,NUM_CLASSES]
    
    @property
    def conv5_3(self):
        # retorna o último Conv2d sem registrá-lo no state_dict
        return self.vgg_feats[28]

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
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_accum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    # concatena todos os batches
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

    model = ASLNetVGG(feature_dim=512).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CFG.LR, momentum=CFG.MOMENTUM)

    from pathlib import Path
    plot_dir = Path("./plots")
    plot_dir.mkdir(exist_ok=True)

    best_val = float('inf')
    for epoch in range(1, CFG.EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion, device)
        vl_loss, vl_acc, vl_preds, vl_labels  = eval_epoch(model, val_dl, criterion, device)
        print(f"[Epoch {epoch:02d}/{CFG.EPOCHS}] "
              f"Train L: {tr_loss:.4f}, A: {tr_acc:.4%} | "
              f"Val L: {vl_loss:.4f}, A: {vl_acc:.4%}")
        
        # gera e salva a matriz a cada época
        _plot_confusion_matrix(vl_preds, vl_labels, plot_dir)
        
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), "filter_static_best.pt")
            print("👉 Best static model saved")

    torch.save(model.state_dict(), "filter_static_final.pt")
    print("👉 Static final model saved")

def inference_with_gradcam(model_path, source=0, target_layer_name='conv5_3'):
    # Inference com Grad-CAM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLNetVGG().to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()
    
    # Preprocessamento
    preprocess = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    # abre webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): 
        return print(f"Erro ao abrir {source}")
    
    # tenta obter pelo nome; se falhar, pega o último Conv2d de vgg_feats
    target_layer = getattr(model, target_layer_name, None)
    if not isinstance(target_layer, nn.Conv2d):
        for m in reversed(model.vgg_feats):
            if isinstance(m, nn.Conv2d):
                target_layer = m
                break
    print(f">>> {target_layer}", "selected as target layer")
    
    while True:
        ret,frame = cap.read()
        if not ret: 
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
        
        # Grad-CAM
        cam = apply_gradcam_torch(model, inp, None, target_layer)
        heatmap = cv2.resize(cam,(frame.shape[1], frame.shape[0]))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4,0)
        
        # predição
        with torch.no_grad():
            pred = model(inp).softmax(1).argmax(1).item()
        label = CFG.LABELS[pred]
        
        #cv2.putText(overlay,f"Pred: {label}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.imshow('Grad-CAM ASL',overlay)
        
        if cv2.waitKey(1) &0xFF == ord('q'): 
            break
        
    cap.release() 
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    inference_with_gradcam("src/model/filters/backup/window_static_99.pt", source=2)
