"""
Módulo de Inferência
===================

Este módulo contém as funcionalidades para inferência em tempo real
e reconhecimento de Libras via webcam.
"""

from .libras_realtime_classifier import LibrasRealtimeClassifier
from .libras_embedded_runtime import EmbeddedPrediction, LibrasEmbeddedRuntime, choose_embedded_prediction
from .prediction_merger import PredictionEvent, PredictionMerger

__all__ = [
	'LibrasRealtimeClassifier',
	'EmbeddedPrediction',
	'LibrasEmbeddedRuntime',
	'choose_embedded_prediction',
	'PredictionEvent',
	'PredictionMerger',
]
