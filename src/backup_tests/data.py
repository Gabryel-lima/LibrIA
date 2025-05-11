import cv2
import os
import numpy as np

""" # Diretórios
input_dir = 'caminho/para/imagens_originais'
output_dir = 'caminho/para/masks_rotuladas'
os.makedirs(output_dir, exist_ok=True)

# Processamento de cada imagem
for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Carrega a imagem
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # Converte para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplica um threshold simples para segmentar
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  
        # 100 é um valor que podemos ajustar conforme seu dataset

        # Normaliza a máscara para 0 e 1
        mask = mask // 255

        # Salva a máscara
        mask_path = os.path.join(output_dir, filename)
        cv2.imwrite(mask_path, mask * 255)  # volta para escala 0-255 para salvar a imagem

print("Máscaras geradas com sucesso!") """

class SegmentationImageDataset:
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        return image, mask
