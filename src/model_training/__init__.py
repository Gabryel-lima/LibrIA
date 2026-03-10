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

__all__ = ['LibrasModelTrainer', 'LibrasLSTMTrainer']
