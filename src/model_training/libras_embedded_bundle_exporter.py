"""Empacota os artefatos embedded em um bundle único para deployment."""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

from config.settings import EMBEDDED_BUNDLE_CONFIG, EMBEDDED_CONFIG, EMBEDDED_TEMPORAL_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERFACES_DIR = PROJECT_ROOT / 'src' / 'interfaces'


def _load_json(path: str) -> Dict[str, object]:
    with open(path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def _ensure_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Artefato embedded não encontrado: {path}')


def _normalize_label_map(metadata: Dict[str, object]) -> List[str]:
    label_map = metadata['label_map']
    return [
        label_map[str(index)] if isinstance(label_map, dict) and str(index) in label_map else label_map[index]
        for index in sorted(int(key) for key in label_map.keys())
    ]


def _bytes_to_cpp_initializer(data: bytes, values_per_line: int = 12) -> str:
    values = [f'0x{value:02x}' for value in data]
    chunks = []
    for start in range(0, len(values), values_per_line):
        chunks.append('    ' + ', '.join(values[start:start + values_per_line]))
    return ',\n'.join(chunks)


def _write_model_array_pair(
    model_path: str,
    header_path: str,
    source_path: str,
    symbol_name: str,
):
    with open(model_path, 'rb') as file_obj:
        model_bytes = file_obj.read()

    header_content = '\n'.join([
        '#pragma once',
        '',
        '#include <cstddef>',
        '',
        f'extern const unsigned char {symbol_name}[];',
        f'extern const std::size_t {symbol_name}_len;',
        '',
    ])

    source_content = '\n'.join([
        f'#include "{os.path.basename(header_path)}"',
        '',
        f'const unsigned char {symbol_name}[] = {{',
        _bytes_to_cpp_initializer(model_bytes),
        '};',
        f'const std::size_t {symbol_name}_len = {len(model_bytes)};',
        '',
    ])

    with open(header_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(header_content)
    with open(source_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(source_content)


def _pico_readme_content(manifest: Dict[str, object], archive_name: str) -> str:
    static_shape = manifest['static']['input_shape']
    temporal_shape = manifest['temporal']['input_shape']
    return '\n'.join([
        '# LibrIA Pico Package',
        '',
        'Pacote gerado automaticamente para exportar o runtime embedded ao RP2040/Pico.',
        '',
        '## Conteudo',
        '',
        '- include/libria_embedded_runtime.h: interface do runtime hibrido.',
        '- include/libria_embedded_bundle_config.h: formas de entrada, labels e thresholds.',
        '- include/libria_embedded_static_model_data.h: declaracao do modelo estatico em C array.',
        '- include/libria_embedded_temporal_model_data.h: declaracao do modelo temporal em C array.',
        '- src/libria_embedded_runtime.cpp: esqueleto TFLite Micro para o dispositivo.',
        '- src/libria_embedded_static_model_data.cpp: bytes do modelo estatico quantizado.',
        '- src/libria_embedded_temporal_model_data.cpp: bytes do modelo temporal quantizado.',
        '- examples/pico_inference_example.cpp: bootstrap de integracao.',
        '- embedded_bundle.json: manifesto completo do bundle.',
        '',
        '## Contrato de entrada',
        '',
        f'- Estatico: {static_shape[0]}x{static_shape[1]} landmarks.',
        f'- Temporal: {temporal_shape[0]}x{temporal_shape[1]} features.',
        '- O host pode continuar usando MediaPipe para gerar o dataset, mas o pacote do Pico nao depende de MediaPipe.',
        '- No dispositivo, basta reproduzir o mesmo ROI controlado e o mesmo layout de landmarks normalizados.',
        '',
        '## Uso rapido',
        '',
        '1. Copie este diretorio ou o archive gerado para o projeto Pico.',
        '2. Adicione os arquivos de include/ e src/ ao build do firmware.',
        '3. Inicialize LibriaEmbeddedRuntime com os arrays gerados.',
        '4. Preencha o TODO do runtime com os interpretadores TFLite Micro e o extrator de ROI local.',
        '',
        f'Archive gerado: {archive_name}',
        '',
    ])


def _pico_example_content() -> str:
    return '\n'.join([
        '#include "libria_embedded_runtime.h"',
        '#include "libria_embedded_static_model_data.h"',
        '#include "libria_embedded_temporal_model_data.h"',
        '',
        '#include <array>',
        '#include <cstdint>',
        '',
        'int main() {',
        '    static std::array<std::uint8_t, 96 * 1024> tensor_arena{};',
        '    static float static_landmarks[21][3] = {};',
        '    static float temporal_window[30][63] = {};',
        '',
        '    LibriaEmbeddedRuntime runtime;',
        '    const bool ok = runtime.Init(',
        '        libria_embedded_static_model_data,',
        '        libria_embedded_static_model_data_len,',
        '        libria_embedded_temporal_model_data,',
        '        libria_embedded_temporal_model_data_len,',
        '        tensor_arena.data(),',
        '        tensor_arena.size()',
        '    );',
        '    if (!ok) {',
        '        return 1;',
        '    }',
        '',
        '    const LibriaEmbeddedPrediction prediction = runtime.PredictHybrid(',
        '        static_landmarks,',
        '        temporal_window,',
        '        true',
        '    );',
        '    return prediction.valid ? 0 : 2;',
        '}',
        '',
    ])


def _pico_cmakelists_content() -> str:
    return '\n'.join([
        'cmake_minimum_required(VERSION 3.13)',
        'project(libria_embedded_runtime C CXX)',
        '',
        'add_library(libria_embedded_runtime STATIC',
        '    src/libria_embedded_runtime.cpp',
        '    src/libria_embedded_static_model_data.cpp',
        '    src/libria_embedded_temporal_model_data.cpp',
        ')',
        '',
        'target_include_directories(libria_embedded_runtime PUBLIC include)',
        'target_compile_features(libria_embedded_runtime PUBLIC cxx_std_17)',
        '',
        '# Integre aqui o pico_sdk_init e a dependencia de TFLite Micro conforme o firmware alvo.',
        '',
    ])


def _build_pico_package(manifest: Dict[str, object]) -> Dict[str, object]:
    package_dir = EMBEDDED_BUNDLE_CONFIG['pico_package_dir']
    include_dir = EMBEDDED_BUNDLE_CONFIG['pico_include_dir']
    src_dir = EMBEDDED_BUNDLE_CONFIG['pico_src_dir']
    examples_dir = EMBEDDED_BUNDLE_CONFIG['pico_examples_dir']

    os.makedirs(include_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)

    runtime_header_name = 'libria_embedded_runtime.h'
    runtime_source_name = 'libria_embedded_runtime.cpp'
    bundle_header_name = os.path.basename(EMBEDDED_BUNDLE_CONFIG['runtime_header_path'])

    shutil.copy2(INTERFACES_DIR / runtime_header_name, os.path.join(include_dir, runtime_header_name))
    shutil.copy2(INTERFACES_DIR / runtime_source_name, os.path.join(src_dir, runtime_source_name))
    shutil.copy2(EMBEDDED_BUNDLE_CONFIG['runtime_header_path'], os.path.join(include_dir, bundle_header_name))

    static_model_path = os.path.join(EMBEDDED_BUNDLE_CONFIG['bundle_dir'], manifest['static']['model_file'])
    temporal_model_path = os.path.join(EMBEDDED_BUNDLE_CONFIG['bundle_dir'], manifest['temporal']['model_file'])

    _write_model_array_pair(
        model_path=static_model_path,
        header_path=os.path.join(include_dir, 'libria_embedded_static_model_data.h'),
        source_path=os.path.join(src_dir, 'libria_embedded_static_model_data.cpp'),
        symbol_name='libria_embedded_static_model_data',
    )
    _write_model_array_pair(
        model_path=temporal_model_path,
        header_path=os.path.join(include_dir, 'libria_embedded_temporal_model_data.h'),
        source_path=os.path.join(src_dir, 'libria_embedded_temporal_model_data.cpp'),
        symbol_name='libria_embedded_temporal_model_data',
    )

    archive_file = (
        EMBEDDED_BUNDLE_CONFIG['pico_archive_path'] +
        '.' + EMBEDDED_BUNDLE_CONFIG.get('pico_archive_format', 'zip')
    )

    with open(EMBEDDED_BUNDLE_CONFIG['pico_readme_path'], 'w', encoding='utf-8') as file_obj:
        file_obj.write(_pico_readme_content(manifest, os.path.basename(archive_file)))
    with open(os.path.join(examples_dir, 'pico_inference_example.cpp'), 'w', encoding='utf-8') as file_obj:
        file_obj.write(_pico_example_content())
    with open(EMBEDDED_BUNDLE_CONFIG['pico_cmake_path'], 'w', encoding='utf-8') as file_obj:
        file_obj.write(_pico_cmakelists_content())

    shutil.copy2(EMBEDDED_BUNDLE_CONFIG['manifest_path'], os.path.join(package_dir, 'embedded_bundle.json'))

    archive_base = EMBEDDED_BUNDLE_CONFIG['pico_archive_path']
    archive_format = EMBEDDED_BUNDLE_CONFIG.get('pico_archive_format', 'zip')
    shutil.make_archive(archive_base, archive_format, package_dir)

    return {
        'package_dir': package_dir,
        'include_dir': include_dir,
        'src_dir': src_dir,
        'examples_dir': examples_dir,
        'archive_file': archive_file,
        'runtime_files': [
            f'include/{runtime_header_name}',
            f'src/{runtime_source_name}',
            f'include/{bundle_header_name}',
            'include/libria_embedded_static_model_data.h',
            'include/libria_embedded_temporal_model_data.h',
            'src/libria_embedded_static_model_data.cpp',
            'src/libria_embedded_temporal_model_data.cpp',
            'examples/pico_inference_example.cpp',
            'embedded_bundle.json',
            'README.md',
            'CMakeLists.txt',
        ],
    }


def _runtime_header_content(manifest: Dict[str, object]) -> str:
    static_labels = manifest['static']['labels']
    temporal_labels = manifest['temporal']['labels']
    priority_labels = manifest['hybrid']['temporal_priority_classes']

    static_labels_cpp = ', '.join(f'"{label}"' for label in static_labels)
    temporal_labels_cpp = ', '.join(f'"{label}"' for label in temporal_labels)
    priority_labels_cpp = ', '.join(f'"{label}"' for label in priority_labels)

    lines = [
        '#pragma once',
        '',
        '// Auto-generated by LibrIA embedded bundle exporter.',
        '#define LIBRIA_EMBEDDED_STATIC_POINTS ' + str(manifest['static']['input_shape'][0]),
        '#define LIBRIA_EMBEDDED_STATIC_CHANNELS ' + str(manifest['static']['input_shape'][1]),
        '#define LIBRIA_EMBEDDED_TEMPORAL_LENGTH ' + str(manifest['temporal']['input_shape'][0]),
        '#define LIBRIA_EMBEDDED_TEMPORAL_FEATURES ' + str(manifest['temporal']['input_shape'][1]),
        '#define LIBRIA_EMBEDDED_STATIC_CLASS_COUNT ' + str(len(static_labels)),
        '#define LIBRIA_EMBEDDED_TEMPORAL_CLASS_COUNT ' + str(len(temporal_labels)),
        '#define LIBRIA_EMBEDDED_PRIORITY_CLASS_COUNT ' + str(len(priority_labels)),
        '#define LIBRIA_EMBEDDED_STATIC_CONF_THRESHOLD ' + str(manifest['hybrid']['static_confidence_threshold']),
        '#define LIBRIA_EMBEDDED_TEMPORAL_CONF_THRESHOLD ' + str(manifest['hybrid']['temporal_confidence_threshold']),
        '',
        'static constexpr const char* kLibriaEmbeddedStaticLabels[] = {' + static_labels_cpp + '};',
        'static constexpr const char* kLibriaEmbeddedTemporalLabels[] = {' + temporal_labels_cpp + '};',
        'static constexpr const char* kLibriaEmbeddedPriorityLabels[] = {' + priority_labels_cpp + '};',
        '',
        '// Input contract: these tensors follow the same landmark layout produced by the host MediaPipe pipeline.',
        '// The embedded runtime does not depend on MediaPipe; the device-side extractor only needs to reproduce',
        '// the same normalized landmark tensor shapes: static (21, 3) and temporal (30, 63) by default.',
    ]
    return '\n'.join(lines) + '\n'


def build_embedded_bundle() -> Dict[str, object]:
    """Cria um bundle único com modelos quantizados e metadados para deployment."""
    static_model_path = EMBEDDED_CONFIG['tflite_model_path']
    static_labels_path = EMBEDDED_CONFIG['label_map_path']
    temporal_model_path = EMBEDDED_TEMPORAL_CONFIG['tflite_model_path']
    temporal_labels_path = EMBEDDED_TEMPORAL_CONFIG['label_map_path']

    for path in [static_model_path, static_labels_path, temporal_model_path, temporal_labels_path]:
        _ensure_file_exists(path)

    static_metadata = _load_json(static_labels_path)
    temporal_metadata = _load_json(temporal_labels_path)

    bundle_dir = EMBEDDED_BUNDLE_CONFIG['bundle_dir']
    os.makedirs(bundle_dir, exist_ok=True)

    static_model_target = os.path.join(bundle_dir, EMBEDDED_BUNDLE_CONFIG['static_model_filename'])
    static_labels_target = os.path.join(bundle_dir, EMBEDDED_BUNDLE_CONFIG['static_labels_filename'])
    temporal_model_target = os.path.join(bundle_dir, EMBEDDED_BUNDLE_CONFIG['temporal_model_filename'])
    temporal_labels_target = os.path.join(bundle_dir, EMBEDDED_BUNDLE_CONFIG['temporal_labels_filename'])

    shutil.copy2(static_model_path, static_model_target)
    shutil.copy2(static_labels_path, static_labels_target)
    shutil.copy2(temporal_model_path, temporal_model_target)
    shutil.copy2(temporal_labels_path, temporal_labels_target)

    static_labels = _normalize_label_map(static_metadata)
    temporal_labels = _normalize_label_map(temporal_metadata)

    manifest = {
        'format_version': 1,
        'bundle_type': 'libria_embedded_hybrid',
        'landmark_contract': {
            'producer': 'host-side MediaPipe-compatible extractor',
            'static_shape': static_metadata.get('input_shape', [21, 3]),
            'temporal_shape': temporal_metadata.get('input_shape', [30, 63]),
            'feature_mode': 'wrist_relative',
            'note': (
                'A coleta do dataset continua igual no host. O bundle embedded nao usa MediaPipe em runtime; '
                'o dispositivo so precisa reproduzir o mesmo layout normalizado de landmarks.'
            ),
        },
        'static': {
            'model_file': EMBEDDED_BUNDLE_CONFIG['static_model_filename'],
            'labels_file': EMBEDDED_BUNDLE_CONFIG['static_labels_filename'],
            'input_shape': static_metadata.get('input_shape', [21, 3]),
            'labels': static_labels,
        },
        'temporal': {
            'model_file': EMBEDDED_BUNDLE_CONFIG['temporal_model_filename'],
            'labels_file': EMBEDDED_BUNDLE_CONFIG['temporal_labels_filename'],
            'input_shape': temporal_metadata.get('input_shape', [30, 63]),
            'labels': temporal_labels,
        },
        'hybrid': {
            'temporal_priority_classes': list(EMBEDDED_TEMPORAL_CONFIG['allowed_classes']),
            'static_confidence_threshold': EMBEDDED_BUNDLE_CONFIG['static_confidence_threshold'],
            'temporal_confidence_threshold': EMBEDDED_BUNDLE_CONFIG['temporal_confidence_threshold'],
        },
    }

    manifest_path = EMBEDDED_BUNDLE_CONFIG['manifest_path']
    with open(manifest_path, 'w', encoding='utf-8') as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=True, indent=2)

    if EMBEDDED_BUNDLE_CONFIG.get('export_runtime_header', False):
        with open(EMBEDDED_BUNDLE_CONFIG['runtime_header_path'], 'w', encoding='utf-8') as file_obj:
            file_obj.write(_runtime_header_content(manifest))

    manifest['pico_package'] = _build_pico_package(manifest)
    with open(manifest_path, 'w', encoding='utf-8') as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=True, indent=2)
    shutil.copy2(manifest_path, os.path.join(EMBEDDED_BUNDLE_CONFIG['pico_package_dir'], 'embedded_bundle.json'))

    return manifest


def main():
    manifest = build_embedded_bundle()
    print(f"Bundle embedded criado em: {EMBEDDED_BUNDLE_CONFIG['bundle_dir']}")
    print(f"Manifesto: {EMBEDDED_BUNDLE_CONFIG['manifest_path']}")
    print(f"Classes estaticas: {manifest['static']['labels']}")
    print(f"Classes temporais: {manifest['temporal']['labels']}")


if __name__ == '__main__':
    main()