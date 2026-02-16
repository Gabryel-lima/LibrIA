"""
Processador de Dataset para Reconhecimento de Libras
===================================================

Este módulo processa as imagens coletadas, extraindo landmarks das mãos
usando MediaPipe e preparando os dados para treinamento do modelo.

Funcionalidades:
- Extração de landmarks das mãos via MediaPipe
- Normalização de coordenadas
- Preparação de features para machine learning
- Serialização do dataset processado
"""

import os
import pickle
import mediapipe as mp
import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple

class LibrasDatasetProcessor:
    """Classe para processamento do dataset de Libras."""
    
    def __init__(self, data_dir: str = './data', output_dir: str = './dataset'):
        """
        Inicializa o processador de dataset.
        
        Args:
            data_dir: Diretório com as imagens coletadas
            output_dir: Diretório para salvar o dataset processado
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Configurar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Inicializar detector de mãos
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, 
            min_detection_confidence=0.3
        )
    
    def process_dataset(self):
        """Processa todo o dataset e salva os dados processados."""
        print("=== LibrIA - Processamento de Dataset ===")
        print("Extraindo landmarks das imagens coletadas...")
        
        data = []
        labels = []
        
        # Processar cada classe
        for class_dir in sorted(os.listdir(self.data_dir)):
            class_path = os.path.join(self.data_dir, class_dir)
            
            if not os.path.isdir(class_path):
                continue
                
            print(f"Processando classe {class_dir}...")
            
            # Processar cada imagem da classe
            for img_filename in os.listdir(class_path):
                img_path = os.path.join(class_path, img_filename)
                
                # Extrair landmarks da imagem
                landmarks = self._extract_landmarks_from_image(img_path)
                
                if landmarks is not None:
                    data.append(landmarks)
                    labels.append(class_dir)
        
        # Salvar dataset processado
        self._save_processed_dataset(data, labels)
        
        print(f"Dataset processado com sucesso!")
        print(f"Total de amostras: {len(data)}")
        print(f"Classes únicas: {len(set(labels))}")
    
    def _extract_landmarks_from_image(self, img_path: str) -> List[float]:
        """
        Extrai landmarks de uma imagem específica.
        
        Args:
            img_path: Caminho para a imagem
            
        Returns:
            Lista de coordenadas normalizadas dos landmarks
        """
        try:
            # Carregar imagem
            img = cv.imread(img_path)
            if img is None:
                return None
                
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            # Processar com MediaPipe
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Pegar apenas a primeira mão detectada
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extrair coordenadas
                x_coords = []
                y_coords = []
                
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                
                # Normalizar coordenadas
                normalized_landmarks = []
                min_x, min_y = min(x_coords), min(y_coords)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x - min_x
                    y = hand_landmarks.landmark[i].y - min_y
                    normalized_landmarks.extend([x, y])
                
                return normalized_landmarks
            
            return None
            
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
            return None
    
    def _save_processed_dataset(self, data: List[List[float]], labels: List[str]):
        """
        Salva o dataset processado em formato pickle.
        
        Args:
            data: Lista de features extraídas
            labels: Lista de labels correspondentes
        """
        # Criar diretório de saída se não existir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Preparar dados para salvamento
        dataset = {
            'data': data,
            'labels': labels,
            'num_features': len(data[0]) if data else 0,
            'num_classes': len(set(labels)) if labels else 0
        }
        
        # Salvar arquivo
        output_path = os.path.join(self.output_dir, 'data.pickle')
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset salvo em: {output_path}")
    
    def get_dataset_info(self) -> Dict:
        """
        Retorna informações sobre o dataset processado.
        
        Returns:
            Dicionário com informações do dataset
        """
        dataset_path = os.path.join(self.output_dir, 'data.pickle')
        
        if not os.path.exists(dataset_path):
            return None
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        return {
            'num_samples': len(dataset['data']),
            'num_features': dataset['num_features'],
            'num_classes': dataset['num_classes'],
            'classes': sorted(set(dataset['labels']))
        }

def main():
    """Função principal para execução do processador de dataset."""
    processor = LibrasDatasetProcessor()
    processor.process_dataset()
    
    # Mostrar informações do dataset
    info = processor.get_dataset_info()
    if info:
        print("\n=== Informações do Dataset ===")
        print(f"Número de amostras: {info['num_samples']}")
        print(f"Número de features: {info['num_features']}")
        print(f"Número de classes: {info['num_classes']}")
        print(f"Classes: {info['classes']}")

if __name__ == "__main__":
    main()
