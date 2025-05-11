import os
import string
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import mediapipe as mp
from tqdm import tqdm  # <-- import do tqdm

# 0) Defina suas classes
labels = list(string.ascii_uppercase) + ["del", "nothing", "space"]
NUM_CLASSES = len(labels)

# 1) Dataset “on-the-fly” com lazy-init do MediaPipe Hands
class LandmarkDataset(Dataset):
    def __init__(self, root_dir, labels, transform=None):
        self.root_dir   = root_dir
        self.labels     = labels
        self.transform  = transform
        self.samples    = []
        for idx, lab in enumerate(labels):
            folder = os.path.join(root_dir, lab)
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg','.png','.jpeg')):
                    self.samples.append((os.path.join(folder, f), idx))
        self.hands = None  # será inicializado no worker

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if self.hands is None:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5)

        path, label = self.samples[i]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)

        res = self.hands.process(img)
        coords = torch.zeros(21*3, dtype=torch.float32)
        if res.multi_hand_landmarks:
            flat = [c for lm in res.multi_hand_landmarks
                         for p in lm.landmark
                             for c in (p.x, p.y, p.z)]
            coords = torch.tensor(flat, dtype=torch.float32)

        return coords, label

# 2) Cria dataset e faz split treino/val
dataset = LandmarkDataset(
    root_dir='/home/gabry/Documentos/projects/LibrIA/data/archive/ASL_Alphabet_Dataset/asl_alphabet_train',
    labels=labels,
    transform=lambda im: cv2.resize(im, (224,224))
)

n_val   = int(0.2 * len(dataset))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

# 3) DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=6, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=6, pin_memory=True)

# 4) Modelo MLP leve
class LandmarkClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21*3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)

device    = torch.device('cpu')
model     = LandmarkClassifier(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5) Loop de treinamento com tqdm
EPOCHS   = 25
best_val = float('inf')

for epoch in range(1, EPOCHS+1):
    # --- Treino ---
    model.train()
    train_loss = 0.0
    train_corr = 0

    loop = tqdm(train_loader,
                desc=f"Epoch {epoch}/{EPOCHS} — train",
                leave=False)
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        train_corr += (logits.argmax(1) == y).sum().item()

        # atualiza a barra com metrics parciais
        loop.set_postfix(loss=loss.item(),
                         acc=(logits.argmax(1)==y).float().mean().item())

    # --- Validação ---
    model.eval()
    val_loss = 0.0
    val_corr = 0

    loop = tqdm(val_loader,
                desc=f"Epoch {epoch}/{EPOCHS} — val  ",
                leave=False)
    with torch.no_grad():
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            val_loss += loss.item() * X.size(0)
            val_corr += (logits.argmax(1) == y).sum().item()

            loop.set_postfix(loss=loss.item(),
                             acc=(logits.argmax(1)==y).float().mean().item())

    # --- Cálculo das métricas finais da época ---
    train_loss /= n_train
    train_acc  = train_corr / n_train
    val_loss   /= n_val
    val_acc    = val_corr / n_val

    print(f"Época {epoch:02d}/{EPOCHS}  "
          f"Train loss={train_loss:.4f} acc={train_acc:.4f}  "
          f"Val loss={val_loss:.4f}   acc={val_acc:.4f}")

    # salva melhor modelo
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "mediapipe_best.pt")
        print("👉 Best static model saved")

# salva ao fim
torch.save(model.state_dict(), "mediapipe_final.pt")
print("👉 Static final model saved")
