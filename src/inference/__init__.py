"""
Módulo de Inferência
===================

Este módulo contém as funcionalidades para inferência em tempo real
e reconhecimento de Libras via webcam.
"""

from .libras_realtime_classifier import LibrasRealtimeClassifier
from .prediction_merger import PredictionEvent, PredictionMerger

__all__ = [
	'LibrasRealtimeClassifier',
	'PredictionEvent',
	'PredictionMerger',
]
