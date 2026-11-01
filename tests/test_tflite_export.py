"""Regressão da exportação TFLite int8 dos trainers embedded.

``TFLiteConverter.from_keras_model`` abortava o processo (``LLVM ERROR: Failed to
infer result type(s)``) com Keras 3 + TensorFlow 2.16. Um abort do runtime C++ não
pode ser capturado em Python e derrubaria a suíte inteira, então a exportação roda
em um subprocesso: uma regressão vira uma falha de teste comum.

Os testes são pulados quando o TensorFlow não está instalado.
"""

import importlib.util
import os
import subprocess
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SCRIPT = """
import os, sys, tempfile
import numpy as np
import tensorflow as tf

kind = sys.argv[1]
tmp = tempfile.mkdtemp()
tflite_path = os.path.join(tmp, 'model.tflite')
common = dict(
    dataset_dir=tmp,
    model_path=os.path.join(tmp, 'model.keras'),
    tflite_path=tflite_path,
    label_map_path=os.path.join(tmp, 'labels.json'),
)
rng = np.random.RandomState(0)

if kind == 'static':
    from src.model_training.libras_embedded_cnn_trainer import LibrasEmbeddedCNNTrainer
    trainer = LibrasEmbeddedCNNTrainer(**common)
    data = rng.rand(20, 21, 3).astype('float32')
    trainer.build_model(num_classes=3)
else:
    from src.model_training.libras_embedded_temporal_cnn_trainer import LibrasEmbeddedTemporalCNNTrainer
    trainer = LibrasEmbeddedTemporalCNNTrainer(**common)
    data = rng.rand(20, 30, 63).astype('float32')
    trainer.build_model(num_classes=2)

trainer._export_tflite_int8(data)

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print('RESULT', np.dtype(inp['dtype']).name, np.dtype(out['dtype']).name,
      list(inp['shape']), list(out['shape']), os.path.getsize(tflite_path) > 0)
"""


def _run_export(kind: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=PROJECT_ROOT, TF_CPP_MIN_LOG_LEVEL='3')
    return subprocess.run(
        [sys.executable, '-c', _SCRIPT, kind],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow não instalado')
class TestTFLiteInt8Export(unittest.TestCase):
    def _assert_export(self, kind: str, expected_result: str):
        result = _run_export(kind)
        self.assertEqual(
            result.returncode, 0,
            f'exportação {kind} falhou (rc={result.returncode}):\n{result.stderr[-1500:]}',
        )
        lines = [line for line in result.stdout.splitlines() if line.startswith('RESULT')]
        self.assertTrue(lines, f'sem RESULT na saída:\n{result.stdout[-500:]}')
        self.assertEqual(lines[-1], expected_result)

    def test_static_export_produces_int8_model(self):
        self._assert_export('static', 'RESULT int8 int8 [1, 21, 3] [1, 3] True')

    def test_temporal_export_produces_int8_model(self):
        self._assert_export('temporal', 'RESULT int8 int8 [1, 30, 63] [1, 2] True')


if __name__ == '__main__':
    unittest.main()
