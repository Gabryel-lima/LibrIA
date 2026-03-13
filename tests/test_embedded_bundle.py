import json
import os
import shutil
import tempfile
import unittest

from config.settings import EMBEDDED_BUNDLE_CONFIG, EMBEDDED_CONFIG, EMBEDDED_TEMPORAL_CONFIG
from src.inference.libras_embedded_runtime import EmbeddedPrediction, choose_embedded_prediction
from src.model_training.libras_embedded_bundle_exporter import build_embedded_bundle


class EmbeddedBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='libria-embedded-bundle-')
        self.original_embedded = dict(EMBEDDED_CONFIG)
        self.original_temporal = dict(EMBEDDED_TEMPORAL_CONFIG)
        self.original_bundle = dict(EMBEDDED_BUNDLE_CONFIG)

        self.static_model_path = os.path.join(self.temp_dir, 'static.tflite')
        self.temporal_model_path = os.path.join(self.temp_dir, 'temporal.tflite')
        self.static_labels_path = os.path.join(self.temp_dir, 'static_labels.json')
        self.temporal_labels_path = os.path.join(self.temp_dir, 'temporal_labels.json')
        self.bundle_dir = os.path.join(self.temp_dir, 'bundle')

        with open(self.static_model_path, 'wb') as file_obj:
            file_obj.write(b'static-model')
        with open(self.temporal_model_path, 'wb') as file_obj:
            file_obj.write(b'temporal-model')

        with open(self.static_labels_path, 'w', encoding='utf-8') as file_obj:
            json.dump(
                {
                    'label_map': {'0': 'A', '1': 'B'},
                    'input_shape': [21, 3],
                },
                file_obj,
            )
        with open(self.temporal_labels_path, 'w', encoding='utf-8') as file_obj:
            json.dump(
                {
                    'label_map': {'0': 'J', '1': 'Z'},
                    'input_shape': [30, 63],
                },
                file_obj,
            )

        EMBEDDED_CONFIG['tflite_model_path'] = self.static_model_path
        EMBEDDED_CONFIG['label_map_path'] = self.static_labels_path
        EMBEDDED_TEMPORAL_CONFIG['tflite_model_path'] = self.temporal_model_path
        EMBEDDED_TEMPORAL_CONFIG['label_map_path'] = self.temporal_labels_path
        EMBEDDED_BUNDLE_CONFIG['bundle_dir'] = self.bundle_dir
        EMBEDDED_BUNDLE_CONFIG['manifest_path'] = os.path.join(self.bundle_dir, 'embedded_bundle.json')
        EMBEDDED_BUNDLE_CONFIG['runtime_header_path'] = os.path.join(self.bundle_dir, 'bundle_config.h')
        EMBEDDED_BUNDLE_CONFIG['pico_package_dir'] = os.path.join(self.bundle_dir, 'pico_package')
        EMBEDDED_BUNDLE_CONFIG['pico_include_dir'] = os.path.join(self.bundle_dir, 'pico_package', 'include')
        EMBEDDED_BUNDLE_CONFIG['pico_src_dir'] = os.path.join(self.bundle_dir, 'pico_package', 'src')
        EMBEDDED_BUNDLE_CONFIG['pico_examples_dir'] = os.path.join(self.bundle_dir, 'pico_package', 'examples')
        EMBEDDED_BUNDLE_CONFIG['pico_cmake_path'] = os.path.join(self.bundle_dir, 'pico_package', 'CMakeLists.txt')
        EMBEDDED_BUNDLE_CONFIG['pico_readme_path'] = os.path.join(self.bundle_dir, 'pico_package', 'README.md')
        EMBEDDED_BUNDLE_CONFIG['pico_archive_path'] = os.path.join(self.bundle_dir, 'libria_embedded_pico_package')
        EMBEDDED_BUNDLE_CONFIG['pico_archive_format'] = 'zip'

    def tearDown(self):
        EMBEDDED_CONFIG.clear()
        EMBEDDED_CONFIG.update(self.original_embedded)
        EMBEDDED_TEMPORAL_CONFIG.clear()
        EMBEDDED_TEMPORAL_CONFIG.update(self.original_temporal)
        EMBEDDED_BUNDLE_CONFIG.clear()
        EMBEDDED_BUNDLE_CONFIG.update(self.original_bundle)
        shutil.rmtree(self.temp_dir)

    def test_build_embedded_bundle_writes_manifest_and_header(self):
        manifest = build_embedded_bundle()

        self.assertEqual(manifest['static']['labels'], ['A', 'B'])
        self.assertEqual(manifest['temporal']['labels'], ['J', 'Z'])
        self.assertTrue(os.path.exists(EMBEDDED_BUNDLE_CONFIG['manifest_path']))
        self.assertTrue(os.path.exists(EMBEDDED_BUNDLE_CONFIG['runtime_header_path']))
        self.assertTrue(os.path.exists(manifest['pico_package']['package_dir']))
        self.assertTrue(os.path.exists(manifest['pico_package']['archive_file']))
        self.assertTrue(
            os.path.exists(
                os.path.join(manifest['pico_package']['include_dir'], 'libria_embedded_static_model_data.h')
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(manifest['pico_package']['src_dir'], 'libria_embedded_temporal_model_data.cpp')
            )
        )

    def test_choose_embedded_prediction_prioritizes_temporal_jz(self):
        static_prediction = EmbeddedPrediction(token='A', confidence=0.99, source='static')
        temporal_prediction = EmbeddedPrediction(token='J', confidence=0.80, source='temporal')

        chosen = choose_embedded_prediction(
            static_prediction=static_prediction,
            temporal_prediction=temporal_prediction,
            static_threshold=0.75,
            temporal_threshold=0.75,
            temporal_priority_classes=['J', 'Z'],
        )

        self.assertEqual(chosen, temporal_prediction)


if __name__ == '__main__':
    unittest.main()