"""Exportação TFLite int8 compartilhada pelos trainers embedded (estático e temporal).

Por que não ``TFLiteConverter.from_keras_model``: com TensorFlow 2.16.x e modelos
Keras 3 esse caminho aborta o processo inteiro com ``LLVM ERROR: Failed to infer
result type(s)``, até sem quantização. Como é um abort do runtime C++, nenhum
``try/except`` do Python o captura. O caminho que funciona é exportar o modelo
como SavedModel (``model.export``) e converter a partir dele.
"""

import os
import tempfile
from typing import Callable, Iterable, List

import numpy as np


def export_int8_tflite(
    model,
    representative_dataset: Callable[[], Iterable[List[np.ndarray]]],
    tflite_path: str,
) -> int:
    """Converte ``model`` (Keras 3) para TFLite int8 completo e grava em ``tflite_path``.

    Entrada e saída ficam em int8, como o runtime embedded espera. Devolve o
    tamanho do arquivo gerado, em bytes.
    """
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as saved_model_dir:
        model.export(saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # convert() precisa rodar antes de o diretório temporário ser apagado.
        tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(tflite_path)), exist_ok=True)
    with open(tflite_path, 'wb') as file_obj:
        file_obj.write(tflite_model)
    return len(tflite_model)
