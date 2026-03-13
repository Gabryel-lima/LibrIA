"""
Módulo de Treinamento de Modelos
===============================

Este módulo contém as funcionalidades para treinamento de modelos
de machine learning para reconhecimento de Libras.
"""

from .libras_model_trainer import LibrasModelTrainer

try:
    from .libras_lstm_trainer import LibrasLSTMTrainer
except (ImportError, RuntimeError):
    LibrasLSTMTrainer = None

try:
    from .libras_embedded_cnn_trainer import LibrasEmbeddedCNNTrainer
except (ImportError, RuntimeError):
    LibrasEmbeddedCNNTrainer = None

try:
    from .libras_embedded_temporal_cnn_trainer import LibrasEmbeddedTemporalCNNTrainer
except (ImportError, RuntimeError):
    LibrasEmbeddedTemporalCNNTrainer = None

try:
    from .libras_embedded_bundle_exporter import build_embedded_bundle
except (ImportError, RuntimeError):
    build_embedded_bundle = None

__all__ = [
    'LibrasModelTrainer',
    'LibrasLSTMTrainer',
    'LibrasEmbeddedCNNTrainer',
    'LibrasEmbeddedTemporalCNNTrainer',
    'build_embedded_bundle',
]
