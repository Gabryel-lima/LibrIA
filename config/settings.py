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
ALPHABET_DICT = {i: chr(65 + i) for i in range(NUMBER_OF_CLASSES)}
HANDS = ['Right', 'Left']

# Configurações do MediaPipe
MEDIAPIPE_CONFIG = {
    'static_image_mode': True,
    'min_detection_confidence': 0.3,
    'min_tracking_confidence': 0.5
}

# Configurações de Processamento
FEATURE_DIMENSIONS = {
    'bounding_box': 42,
    'wrist_relative': 63,
}
FEATURE_MODE = 'wrist_relative'
FEATURE_DIMENSION = FEATURE_DIMENSIONS[FEATURE_MODE]
FEATURE_SIZE = FEATURE_DIMENSION

# Configurações de calibração de câmera
CAMERA_CONFIG = {
    'enabled': True,
    'camera_matrix_path': './config/camera_matrix.npy',
    'dist_coeffs_path': './config/dist_coeffs.npy',
}

# Configurações do modelo temporal
SEQUENCES_DIR = './dataset/sequences'
LSTM_CONFIG = {
    'sequence_length': 30,
    'feature_size': FEATURE_DIMENSION,
    'num_classes': NUMBER_OF_CLASSES,
    'epochs': 50,
    'batch_size': 16,
    'validation_split': 0.2,
    'model_path': './model/libras_lstm.keras',
    'label_map_path': './model/libras_lstm_labels.pickle',
}

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
    directories = [DATA_DIR, DATASET_DIR, MODEL_DIR, OUTPUT_DIR, SEQUENCES_DIR]
    
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
    
    if FEATURE_MODE not in FEATURE_DIMENSIONS:
        errors.append(f"FEATURE_MODE inválido: {FEATURE_MODE}")
    elif FEATURE_DIMENSION != FEATURE_DIMENSIONS[FEATURE_MODE]:
        errors.append(
            "FEATURE_DIMENSION deve corresponder ao FEATURE_MODE configurado"
        )
    
    if len(ALPHABET_DICT) != 26:
        errors.append("ALPHABET_DICT deve conter 26 letras do alfabeto completo")
    
    return errors
