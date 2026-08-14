"""
Módulo de Configurações
======================

Este módulo contém todas as configurações centralizadas do projeto.
"""

from .settings import *

__all__ = [
    'DATA_DIR', 'DATASET_DIR', 'MODEL_DIR', 'OUTPUT_DIR',
    'STATIC_DATASET_DIR', 'TEMPORAL_DATASET_DIR',
    'DATASET_SIZE', 'NUMBER_OF_CLASSES', 'ALPHABET_DICT', 'HANDS',
    'STATIC_LABELS', 'TEMPORAL_LABELS', 'LEXICAL_LABELS', 'FUNCTIONAL_LABELS',
    'TEMPORAL_VOCABULARY_LABELS',
    'MEDIAPIPE_CONFIG', 'FEATURE_DIMENSIONS', 'FEATURE_MODE',
    'FEATURE_DIMENSION', 'FEATURE_SIZE', 'CAMERA_CONFIG', 'COLLECTION_CONFIG',
    'LSTM_CONFIG', 'EMBEDDED_CONFIG', 'EMBEDDED_TEMPORAL_CONFIG', 'EMBEDDED_BUNDLE_CONFIG',
    'MODEL_CONFIG', 'TRAINING_CONFIG', 'INFERENCE_CONFIG', 'HYBRID_INFERENCE_CONFIG',
    'TEMPORAL_PIPELINE_CONFIG', 'EVALUATION_CONFIG', 'UI_CONFIG',
    'LOGGING_CONFIG', 'PERFORMANCE_CONFIG',
    'create_directories', 'get_alphabet_mapping', 'get_class_names',
    'get_num_classes', 'validate_config'
]
