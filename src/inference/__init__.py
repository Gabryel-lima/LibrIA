"""
Módulo de Inferência
===================

Este módulo contém as funcionalidades para inferência em tempo real
e reconhecimento de Libras via webcam.
"""

from .libras_realtime_classifier import LibrasRealtimeClassifier
from .libras_embedded_runtime import EmbeddedPrediction, LibrasEmbeddedRuntime, choose_embedded_prediction
from .prediction_merger import PredictionEvent, PredictionMerger
from .motion_detector import MotionDetector
from .prediction_smoother import DuplicateSuppressor, ProbabilitySmoother
from .sign_segmenter import SignSegment, SignSegmenter
from .sign_token import SignToken, build_rejected_token, build_token
from .temporal_buffer import TemporalBuffer, resample_sequence
from .temporal_pipeline import TemporalPipeline

__all__ = [
	'LibrasRealtimeClassifier',
	'EmbeddedPrediction',
	'LibrasEmbeddedRuntime',
	'choose_embedded_prediction',
	'PredictionEvent',
	'PredictionMerger',
	'MotionDetector',
	'DuplicateSuppressor',
	'ProbabilitySmoother',
	'SignSegment',
	'SignSegmenter',
	'SignToken',
	'build_token',
	'build_rejected_token',
	'TemporalBuffer',
	'resample_sequence',
	'TemporalPipeline',
]
