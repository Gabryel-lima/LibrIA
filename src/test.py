import cv2
import numpy as np
import os
from data import SegmentationImageDataset

def test_dataset():
    # Diretórios de entrada
    image_dir = 'caminho/para/imagens_originais'
    mask_dir = 'caminho/para/masks_rotuladas'

    # Cria o dataset
    dataset = SegmentationImageDataset(image_dir, mask_dir)

    # Testa o tamanho do dataset
    assert len(dataset) > 0, "O dataset está vazio!"

    # Testa o carregamento de uma imagem e máscara
    image, mask = dataset[0]
    
    # Verifica se a imagem e a máscara têm o mesmo tamanho
    assert image.shape[:2] == mask.shape[:2], "A imagem e a máscara não têm o mesmo tamanho!"

    print("Teste do dataset passou com sucesso!")
    
if __name__ == "__main__":
    #test_dataset()
    import torch
    from GestureNet import ASLNet

    model = ASLNet().to("cpu")  # Teste com CPU primeiro
    dummy_input = torch.randn(1, 1, 32, 32)
    output = model(dummy_input)
    print("Output OK:", output.shape)



