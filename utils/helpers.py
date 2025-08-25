"""
Utilitários e Funções Auxiliares para LibrIA
============================================

Este módulo contém funções utilitárias e auxiliares utilizadas
em todo o projeto LibrIA.
"""

import os
import cv2 as cv
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

def setup_logging(log_file: str = 'libras.log', level: str = 'INFO'):
    """
    Configura o sistema de logging.
    
    Args:
        log_file: Arquivo para salvar os logs
        level: Nível de logging
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_model(model_path: str):
    """
    Carrega um modelo salvo em pickle.
    
    Args:
        model_path: Caminho para o arquivo do modelo
        
    Returns:
        Modelo carregado
    """
    try:
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
            return model_dict['model']
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo: {e}")

def save_model(model: Any, model_path: str, metadata: Dict = None):
    """
    Salva um modelo em formato pickle.
    
    Args:
        model: Modelo a ser salvo
        model_path: Caminho para salvar o modelo
        metadata: Metadados adicionais para salvar
    """
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Preparar dados para salvar
        save_data = {'model': model}
        if metadata:
            save_data.update(metadata)
        
        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)
            
    except Exception as e:
        raise RuntimeError(f"Erro ao salvar modelo: {e}")

def load_dataset(dataset_path: str):
    """
    Carrega um dataset salvo em pickle.
    
    Args:
        dataset_path: Caminho para o arquivo do dataset
        
    Returns:
        Dataset carregado
    """
    try:
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar dataset: {e}")

def save_dataset(data: List, labels: List, dataset_path: str, metadata: Dict = None):
    """
    Salva um dataset em formato pickle.
    
    Args:
        data: Dados do dataset
        labels: Labels do dataset
        dataset_path: Caminho para salvar o dataset
        metadata: Metadados adicionais
    """
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        # Preparar dados para salvar
        dataset = {
            'data': data,
            'labels': labels,
            'num_features': len(data[0]) if data else 0,
            'num_classes': len(set(labels)) if labels else 0
        }
        
        if metadata:
            dataset.update(metadata)
        
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
            
    except Exception as e:
        raise RuntimeError(f"Erro ao salvar dataset: {e}")

def extract_hand_landmarks(image: np.ndarray, hands_detector) -> Optional[List[float]]:
    """
    Extrai landmarks de uma mão de uma imagem.
    
    Args:
        image: Imagem em formato RGB
        hands_detector: Detector de mãos do MediaPipe
        
    Returns:
        Lista de coordenadas normalizadas dos landmarks
    """
    try:
        results = hands_detector.process(image)
        
        if results.multi_hand_landmarks:
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
        logging.error(f"Erro ao extrair landmarks: {e}")
        return None

def draw_hand_landmarks(frame: np.ndarray, hand_landmarks, mp_drawing, mp_hands, mp_drawing_styles):
    """
    Desenha landmarks da mão no frame.
    
    Args:
        frame: Frame da webcam
        hand_landmarks: Landmarks da mão
        mp_drawing: Módulo de desenho do MediaPipe
        mp_hands: Módulo de mãos do MediaPipe
        mp_drawing_styles: Estilos de desenho do MediaPipe
        
    Returns:
        Frame com landmarks desenhados
    """
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    return frame

def calculate_bounding_box(hand_landmarks, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """
    Calcula a bounding box da mão.
    
    Args:
        hand_landmarks: Landmarks da mão
        frame_shape: Formato do frame (H, W, C)
        
    Returns:
        Tupla com coordenadas da bounding box (x1, y1, x2, y2)
    """
    H, W, _ = frame_shape
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
    
    x1 = int(min(x_coords) * W) - 10
    y1 = int(min(y_coords) * H) - 10
    x2 = int(max(x_coords) * W) + 10
    y2 = int(max(y_coords) * H) + 10
    
    return x1, y1, x2, y2

def draw_prediction_overlay(frame: np.ndarray, prediction: str, confidence: float,
                          bbox: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 255, 0)):
    """
    Desenha overlay com a predição no frame.
    
    Args:
        frame: Frame da webcam
        prediction: Predição (letra)
        confidence: Confiança da predição
        bbox: Bounding box (x1, y1, x2, y2)
        color: Cor do overlay
        
    Returns:
        Frame com overlay
    """
    x1, y1, x2, y2 = bbox
    
    # Desenhar retângulo
    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Desenhar texto da predição
    text = f"{prediction} ({confidence:.2f})"
    cv.putText(frame, text, (x1, y1 - 10), 
              cv.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv.LINE_AA)
    
    return frame

def add_info_overlay(frame: np.ndarray, title: str, instructions: str = ""):
    """
    Adiciona informações gerais no frame.
    
    Args:
        frame: Frame da webcam
        title: Título a ser exibido
        instructions: Instruções a serem exibidas
        
    Returns:
        Frame com informações
    """
    # Adicionar título
    cv.putText(frame, title, (10, 30), 
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Adicionar instruções se fornecidas
    if instructions:
        cv.putText(frame, instructions, (10, frame.shape[0] - 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def save_screenshot(frame: np.ndarray, prefix: str = "screenshot") -> str:
    """
    Salva um screenshot do frame atual.
    
    Args:
        frame: Frame a ser salvo
        prefix: Prefixo do nome do arquivo
        
    Returns:
        Nome do arquivo salvo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    cv.imwrite(filename, frame)
    return filename

def setup_video_recording(cap, output_path: str, fps: float = 20.0):
    """
    Configura a gravação de vídeo.
    
    Args:
        cap: Objeto de captura de vídeo
        output_path: Caminho para salvar o vídeo
        fps: Frames por segundo
        
    Returns:
        Objeto VideoWriter
    """
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    # Determinar o codec baseado na extensão do arquivo
    if output_path.lower().endswith('.mp4'):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
    elif output_path.lower().endswith('.avi'):
        fourcc = cv.VideoWriter_fourcc(*'XVID')
    else:
        # Padrão para MP4
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
    
    return cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def validate_image_path(image_path: str) -> bool:
    """
    Valida se um caminho de imagem existe e é válido.
    
    Args:
        image_path: Caminho para a imagem
        
    Returns:
        True se válido, False caso contrário
    """
    if not os.path.exists(image_path):
        return False
    
    # Tentar carregar a imagem
    try:
        img = cv.imread(image_path)
        return img is not None
    except:
        return False

def get_file_size_mb(file_path: str) -> float:
    """
    Obtém o tamanho de um arquivo em MB.
    
    Args:
        file_path: Caminho para o arquivo
        
    Returns:
        Tamanho em MB
    """
    if not os.path.exists(file_path):
        return 0.0
    
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def format_time(seconds: float) -> str:
    """
    Formata tempo em segundos para string legível.
    
    Args:
        seconds: Tempo em segundos
        
    Returns:
        String formatada
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"
