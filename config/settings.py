"""
Configurações do Projeto LibrIA
==============================

Este módulo contém todas as configurações centralizadas do projeto,
incluindo parâmetros de coleta de dados, processamento e inferência.
"""

import os
from typing import Dict, List

# Configurações de Diretórios
DATA_DIR = './data'
DATASET_DIR = './dataset'
MODEL_DIR = './model'
OUTPUT_DIR = './output'

# Configurações de Coleta de Dados
DATASET_SIZE = 150  # Imagens por classe
NUMBER_OF_CLASSES = 26
ALPHABET_DICT = {i: chr(65 + i) for i in range(NUMBER_OF_CLASSES) 
                 if chr(65 + i) not in ['J', 'Z']}
HANDS = ['Right', 'Left']

# Configurações do MediaPipe
MEDIAPIPE_CONFIG = {
    'static_image_mode': True,
    'min_detection_confidence': 0.3,
    'min_tracking_confidence': 0.5
}

# Configurações de Processamento
FEATURE_DIMENSION = 42  # 21 landmarks × 2 coordenadas (x, y)

# Configurações do Modelo
MODEL_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1  # Usar todos os cores disponíveis
}

# Configurações de Treinamento
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Configurações de Inferência
INFERENCE_CONFIG = {
    'min_detection_confidence': 0.3,
    'prediction_interval': 20,  # Frames entre predições
    'record_video': True,
    'output_video_path': 'output.mp4'
}

# Configurações de Interface
UI_CONFIG = {
    'window_title': 'LibrIA - Reconhecimento de Libras',
    'font_scale': 1.3,
    'font_thickness': 3,
    'bbox_color': (0, 255, 0),  # Verde
    'text_color': (0, 255, 0),  # Verde
    'info_color': (255, 255, 255),  # Branco
    'instruction_color': (200, 200, 200)  # Cinza claro
}

# Configurações de Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'libras.log'
}

# Configurações de Performance
PERFORMANCE_CONFIG = {
    'max_fps': 30,
    'frame_delay': 25,  # ms
    'memory_limit': 1024  # MB
}

def create_directories():
    """Cria todos os diretórios necessários para o projeto."""
    directories = [DATA_DIR, DATASET_DIR, MODEL_DIR, OUTPUT_DIR]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório criado: {directory}")

def get_alphabet_mapping() -> Dict[int, str]:
    """Retorna o mapeamento de classes para letras do alfabeto."""
    return ALPHABET_DICT

def get_class_names() -> List[str]:
    """Retorna a lista de nomes das classes."""
    return list(ALPHABET_DICT.values())

def get_num_classes() -> int:
    """Retorna o número total de classes."""
    return len(ALPHABET_DICT)

def validate_config():
    """Valida as configurações do projeto."""
    errors = []
    
    # Validar diretórios
    if not os.path.exists(DATA_DIR):
        errors.append(f"Diretório de dados não encontrado: {DATA_DIR}")
    
    # Validar parâmetros
    if DATASET_SIZE <= 0:
        errors.append("DATASET_SIZE deve ser maior que zero")
    
    if FEATURE_DIMENSION != 42:
        errors.append("FEATURE_DIMENSION deve ser 42 (21 landmarks × 2 coordenadas)")
    
    if len(ALPHABET_DICT) != 24:
        errors.append("ALPHABET_DICT deve conter 24 letras (excluindo J e Z)")
    
    return errors
