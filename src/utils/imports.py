from typing import (
    TypeVar,
    Union, 
    Generic,
    Iterable
) # -> typing module

import json
import traceback
import numpy as np
import pandas as pd
#from enum import Enum #TODO: Por enquanto vou manter os types aqui mesmo, até decidir se vou utilizar o file data_types
# from types import NoneType

# Tentar importar TensorFlow e Keras (requerem suporte AVX)
TENSORFLOW_AVAILABLE = False
KERAS_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"⚠️  TensorFlow não disponível: {type(e).__name__}")
    print("   → CPU pode não suportar AVX")
    # Criar stub
    class TFStub:
        pass
    tf = TFStub()

try:
    from keras import KerasTensor, layers, models, Model
    from keras.src.utils.numerical_utils import to_categorical
    KERAS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"⚠️  Keras não disponível: {type(e).__name__}")
    print("   → Dependência do TensorFlow")
    # Criar stubs
    class KerasTensor:
        pass
    layers = None
    models = None
    Model = None
    def to_categorical(*args, **kwargs):
        raise RuntimeError("Keras não disponível - to_categorical não pode ser usado")

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

__all__ = [
    'TypeVar',
    'Union',
    'Generic',
    'Iterable',
    'json',
    'traceback',
    'np',
    'pd',
    'tf',
    'KerasTensor',
    'plt',
    'layers',
    'models',
    'Model',
    'train_test_split',
    'to_categorical',
    'TENSORFLOW_AVAILABLE',
    'KERAS_AVAILABLE'
]
