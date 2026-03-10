"""
Módulo de Inferência
===================

Este módulo contém as funcionalidades para inferência em tempo real
e reconhecimento de Libras via webcam.
"""

from .libras_realtime_classifier import LibrasRealtimeClassifier

try:
	from .libras_lstm_realtime_classifier import LibrasLSTMRealtimeClassifier
except (ImportError, RuntimeError):
	LibrasLSTMRealtimeClassifier = None

__all__ = ['LibrasRealtimeClassifier', 'LibrasLSTMRealtimeClassifier']
