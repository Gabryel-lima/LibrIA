"""
Módulo de Configurações
======================

Este módulo contém todas as configurações centralizadas do projeto.
"""

from .settings import *

__all__ = [
    'DATA_DIR', 'DATASET_DIR', 'MODEL_DIR', 'OUTPUT_DIR',
    'DATASET_SIZE', 'NUMBER_OF_CLASSES', 'ALPHABET_DICT', 'HANDS',
    'MEDIAPIPE_CONFIG', 'FEATURE_DIMENSIONS', 'FEATURE_MODE',
    'FEATURE_DIMENSION', 'FEATURE_SIZE', 'CAMERA_CONFIG', 'SEQUENCES_DIR',
    'LSTM_CONFIG', 'MODEL_CONFIG', 'TRAINING_CONFIG', 'INFERENCE_CONFIG', 'UI_CONFIG',
    'LOGGING_CONFIG', 'PERFORMANCE_CONFIG',
    'create_directories', 'get_alphabet_mapping', 'get_class_names',
    'get_num_classes', 'validate_config'
]
