"""
Configurações do Projeto LibrIA
==============================

Este módulo contém todas as configurações centralizadas do projeto,
incluindo parâmetros de coleta de dados, processamento e inferência.
"""

import os
from typing import Dict, List

from config.vocabulary import (
    MODALITY_STATIC,
    MODALITY_TEMPORAL,
    SIGN_TYPE_ALPHABET,
    SIGN_TYPE_FUNCTIONAL,
    SIGN_TYPE_LEXICAL,
    UNKNOWN_LABEL,
    get_labels,
    validate_vocabulary,
)

# Configurações de Diretórios
DATA_DIR = './data'
DATASET_DIR = './dataset'
MODEL_DIR = './model'
OUTPUT_DIR = './output'
STATIC_DATASET_DIR = os.path.join(DATASET_DIR, 'static')
TEMPORAL_DATASET_DIR = os.path.join(DATASET_DIR, 'temporal')
LEGACY_STATIC_DATASET_PATH = os.path.join(DATASET_DIR, 'data.pickle')

# Configurações de Coleta de Dados
DATASET_SIZE = int(os.getenv('LIBRIA_DATASET_SIZE', '150'))
NUMBER_OF_CLASSES = 26
ALPHABET_DICT = {i: chr(65 + i) for i in range(NUMBER_OF_CLASSES)}
HANDS = ['Right', 'Left']

# Labels derivados do vocabulário (config/vocabulary.py é a fonte de verdade).
# STATIC_LABELS/TEMPORAL_LABELS continuam sendo apenas o alfabeto manual para
# não alterar os fluxos de coleta e treino existentes; o vocabulário completo
# (palavras e gestos funcionais) fica em TEMPORAL_VOCABULARY_LABELS.
STATIC_LABELS = get_labels(sign_type=SIGN_TYPE_ALPHABET, modality=MODALITY_STATIC)
TEMPORAL_LABELS = get_labels(sign_type=SIGN_TYPE_ALPHABET, modality=MODALITY_TEMPORAL)
LEXICAL_LABELS = get_labels(sign_type=SIGN_TYPE_LEXICAL)
FUNCTIONAL_LABELS = get_labels(sign_type=SIGN_TYPE_FUNCTIONAL)
TEMPORAL_VOCABULARY_LABELS = get_labels(modality=MODALITY_TEMPORAL)

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

# Configurações de coleta
COLLECTION_CONFIG = {
    'static_samples_per_label': 30,
    'temporal_samples_per_label': 30,
    'min_detection_confidence': 0.8,
    'min_tracking_confidence': 0.8,
    'static_frame_ext': '.png',
    # Metadados gravados junto de cada amostra (ver src/dataset/sample_metadata.py).
    # Sobrescreva pela linha de comando a cada sessão de coleta.
    'write_sample_metadata': True,
    'default_subject_id': 'desconhecido',
    'default_camera_id': 'desconhecido',
    'default_environment': 'desconhecido',
    'default_dominant_hand': 'desconhecido',
}

# Avaliação por pessoa, por classe e por ambiente (Fase 1 do plano).
EVALUATION_CONFIG = {
    'split_ratios': {'train': 0.6, 'validation': 0.2, 'test': 0.2},
    'unknown_label': UNKNOWN_LABEL,
    # Abaixo deste limiar a predição vira UNKNOWN_LABEL em vez de virar tradução.
    'rejection_threshold': 0.75,
    'report_path': './output/evaluation_report.json',
}

LSTM_CONFIG = {
    'sequence_length': 30,
    'feature_size': FEATURE_DIMENSION,
    'num_classes': NUMBER_OF_CLASSES,
    'epochs': 50,
    'batch_size': 16,
    'validation_split': 0.2,
    'model_path': './model/libras_lstm.keras',
    'label_map_path': './model/libras_lstm_labels.pickle',
    'allowed_classes': ['J', 'Z'],
    'restrict_to_allowed_classes': True,
    # Ao ampliar allowed_classes para o vocabulário lexical, deixe False para
    # treinar com as classes já coletadas em vez de falhar nas que faltam.
    'require_all_allowed_classes': True,
    'confidence_threshold': 0.85,
}

EMBEDDED_CONFIG = {
    'dataset_dir': STATIC_DATASET_DIR,
    'input_points': 21,
    'input_channels': 3,
    'epochs': 20,
    'batch_size': 16,
    'validation_split': 0.2,
    'min_samples_per_class': 8,
    'keras_model_path': './model/libria_embedded_cnn.keras',
    'tflite_model_path': './model/libria_embedded_cnn_int8.tflite',
    'label_map_path': './model/libria_embedded_cnn_labels.json',
}

EMBEDDED_TEMPORAL_CONFIG = {
    'dataset_dir': TEMPORAL_DATASET_DIR,
    'sequence_length': LSTM_CONFIG['sequence_length'],
    'feature_size': FEATURE_DIMENSION,
    'epochs': 25,
    'batch_size': 8,
    'validation_split': 0.2,
    'allowed_classes': ['J', 'Z'],
    'keras_model_path': './model/libria_embedded_temporal_cnn.keras',
    'tflite_model_path': './model/libria_embedded_temporal_cnn_int8.tflite',
    'label_map_path': './model/libria_embedded_temporal_cnn_labels.json',
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
    'output_video_path': 'output.mp4',
    'static_confidence_threshold': 0.75,
}

HYBRID_INFERENCE_CONFIG = {
    'temporal_priority_classes': tuple(LSTM_CONFIG['allowed_classes']),
    'temporal_confidence_threshold': LSTM_CONFIG['confidence_threshold'],
    'static_confidence_threshold': INFERENCE_CONFIG['static_confidence_threshold'],
    'prediction_cooldown_seconds': 0.5,
    'window_overlap_tolerance_seconds': 0.25,
    'overlay_history_size': 5,
}

# Pipeline temporal robusto (Fase 2): buffer, movimento, segmentação,
# suavização e supressão de duplicatas. Ver src/inference/temporal_pipeline.py.
TEMPORAL_PIPELINE_CONFIG = {
    # Detecção de movimento (energia média por landmark, entre 0 e ~1).
    'motion_smoothing': 0.6,
    'motion_start_threshold': 0.012,
    'motion_end_threshold': 0.006,  # menor que o de início: histerese.
    # Segmentação
    'min_start_frames': 2,
    'min_end_frames': 5,
    'min_segment_frames': 8,
    'min_duration_seconds': 0.2,
    'max_duration_seconds': 4.0,
    'max_absent_frames': 5,
    # Suavização e deduplicação
    'smoothing_window': 5,
    'duplicate_window_seconds': 1.0,
    # Emissão
    'emit_partial_tokens': True,
    'partial_interval_frames': 5,
    'temporal_confidence_threshold': LSTM_CONFIG['confidence_threshold'],
    'static_confidence_threshold': INFERENCE_CONFIG['static_confidence_threshold'],
    'static_interval_frames': INFERENCE_CONFIG['prediction_interval'],
    # Após um sinal temporal, a mão volta ao repouso passando por poses que o
    # modelo estático leria como letras. Silencia o fallback nesse intervalo.
    'static_cooldown_seconds': HYBRID_INFERENCE_CONFIG['prediction_cooldown_seconds'],
}

EMBEDDED_BUNDLE_CONFIG = {
    'bundle_dir': './model/embedded_bundle',
    'manifest_path': './model/embedded_bundle/embedded_bundle.json',
    'runtime_header_path': './model/embedded_bundle/libria_embedded_bundle_config.h',
    'pico_package_dir': './model/embedded_bundle/pico_package',
    'pico_include_dir': './model/embedded_bundle/pico_package/include',
    'pico_src_dir': './model/embedded_bundle/pico_package/src',
    'pico_examples_dir': './model/embedded_bundle/pico_package/examples',
    'pico_cmake_path': './model/embedded_bundle/pico_package/CMakeLists.txt',
    'pico_readme_path': './model/embedded_bundle/pico_package/README.md',
    'pico_archive_path': './model/embedded_bundle/libria_embedded_pico_package',
    'pico_archive_format': 'zip',
    'static_model_filename': 'libria_embedded_cnn_int8.tflite',
    'static_labels_filename': 'libria_embedded_cnn_labels.json',
    'temporal_model_filename': 'libria_embedded_temporal_cnn_int8.tflite',
    'temporal_labels_filename': 'libria_embedded_temporal_cnn_labels.json',
    'static_confidence_threshold': 0.75,
    'temporal_confidence_threshold': 0.75,
    'export_runtime_header': True,
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
    directories = [
        DATA_DIR,
        DATASET_DIR,
        STATIC_DATASET_DIR,
        TEMPORAL_DATASET_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
    ]
    
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

    errors.extend(validate_vocabulary())

    split_total = sum(EVALUATION_CONFIG['split_ratios'].values())
    if abs(split_total - 1.0) > 1e-6:
        errors.append("EVALUATION_CONFIG['split_ratios'] deve somar 1.0")

    return errors
