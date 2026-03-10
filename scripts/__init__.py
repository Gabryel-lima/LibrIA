"""
Módulo de Scripts
================

Este módulo contém os scripts de utilitários para calibração de câmera,
processamento de imagens e outras tarefas auxiliares do projeto LibrIA.
"""

import importlib

__all__ = [
    'calibrate_camera',
    'expand_image_patterns',
    'calibrate_main',
    'collect_sequences',
    'preprocess_frame',
    'extract_landmarks_by_mode',
    'collect_sequences_main',
]


def __getattr__(name):
    if name in {'calibrate_camera', 'expand_image_patterns', 'calibrate_main'}:
        module = importlib.import_module('.calibrate_camera', __name__)
        exports = {
            'calibrate_camera': module.calibrate_camera,
            'expand_image_patterns': module.expand_image_patterns,
            'calibrate_main': module.main,
        }
        return exports[name]

    if name in {
        'collect_sequences',
        'preprocess_frame',
        'extract_landmarks_by_mode',
        'collect_sequences_main',
    }:
        module = importlib.import_module('.collect_sequences', __name__)
        exports = {
            'collect_sequences': module.collect_sequences,
            'preprocess_frame': module.preprocess_frame,
            'extract_landmarks_by_mode': module.extract_landmarks_by_mode,
            'collect_sequences_main': module.main,
        }
        return exports[name]

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
