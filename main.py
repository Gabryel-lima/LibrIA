#!/usr/bin/env python3
"""
LibrIA — interface de linha de comando
======================================

Ponto de entrada único para coleta, treino e inferência. Os nomes dos comandos
são idênticos aos alvos do Makefile, de modo que `make train-temporal` e
`python main.py train-temporal` fazem exatamente a mesma coisa.

    python main.py --help
"""

import argparse
import os
import sys
from pathlib import Path

# Configurar o path do projeto para importações corretas
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import (
    COLLECTION_CONFIG,
    EMBEDDED_BUNDLE_CONFIG,
    EMBEDDED_CONFIG,
    EMBEDDED_TEMPORAL_CONFIG,
    FUNCTIONAL_LABELS,
    LEGACY_STATIC_DATASET_PATH,
    LEXICAL_LABELS,
    LSTM_CONFIG,
    STATIC_DATASET_DIR,
    STATIC_LABELS,
    TEMPORAL_DATASET_DIR,
    TEMPORAL_LABELS,
    create_directories,
    validate_config,
)
from config.vocabulary import UNKNOWN_LABEL
from scripts.collect_dataset import CaptureContext, collect_static, collect_temporal
from src.inference.libras_embedded_runtime import LibrasEmbeddedRuntime
from src.inference.libras_hybrid_realtime_classifier import LibrasHybridRealtimeClassifier
from src.inference.libras_lstm_realtime_classifier import LibrasLSTMRealtimeClassifier
from src.inference.libras_realtime_classifier import LibrasRealtimeClassifier
from src.model_training.libras_embedded_bundle_exporter import build_embedded_bundle
from src.model_training.libras_embedded_cnn_trainer import LibrasEmbeddedCNNTrainer
from src.model_training.libras_embedded_temporal_cnn_trainer import LibrasEmbeddedTemporalCNNTrainer
from src.model_training.libras_lstm_trainer import LibrasLSTMTrainer
from src.model_training.libras_model_trainer import LibrasModelTrainer
from utils.helpers import setup_logging

STATIC_MODEL_PATH = './model/model.pickle'


# ---------------------------------------------------------------------------
# Verificações de pré-requisitos
# ---------------------------------------------------------------------------

def _has_static_dataset() -> bool:
    return os.path.exists(LEGACY_STATIC_DATASET_PATH) or os.path.isdir(STATIC_DATASET_DIR)


def _has_temporal_dataset() -> bool:
    return os.path.isdir(TEMPORAL_DATASET_DIR)


def _prepare_runtime() -> bool:
    """Valida a configuração e cria os diretórios do projeto."""
    errors = validate_config()
    if errors:
        print('Erros de configuração encontrados:')
        for error in errors:
            print(f'  - {error}')
        return False

    create_directories()
    return True


def _capture_context(args) -> CaptureContext:
    return CaptureContext(
        subject_id=args.subject,
        camera_id=args.camera_id,
        environment=args.environment,
        dominant_hand=args.dominant_hand,
    )


# ---------------------------------------------------------------------------
# Coleta
# ---------------------------------------------------------------------------

def collect_static_data(args) -> bool:
    """Coleta o alfabeto manual estático em dataset/static."""
    print('=== Coleta estática ===')
    if not _prepare_runtime():
        return False

    try:
        collect_static(
            labels=STATIC_LABELS,
            samples_per_label=args.samples,
            output_dir=STATIC_DATASET_DIR,
            camera_index=args.camera_index,
            context=_capture_context(args),
        )
        print('✅ Coleta estática concluída!')
        return True
    except Exception as error:
        print(f'❌ Erro durante a coleta estática: {error}')
        return False


def _collect_temporal_labels(args, labels, description: str) -> bool:
    print(f'=== Coleta temporal — {description} ===')
    if not _prepare_runtime():
        return False

    try:
        collect_temporal(
            labels=labels,
            num_sequences=args.sequences,
            seq_length=LSTM_CONFIG['sequence_length'],
            output_dir=TEMPORAL_DATASET_DIR,
            camera_index=args.camera_index,
            context=_capture_context(args),
        )
        print('✅ Coleta temporal concluída!')
        return True
    except Exception as error:
        print(f'❌ Erro durante a coleta temporal: {error}')
        return False


def collect_temporal_data(args) -> bool:
    """Coleta as letras temporais (J e Z)."""
    return _collect_temporal_labels(args, TEMPORAL_LABELS, 'alfabeto (J, Z)')


def collect_words(args) -> bool:
    """Coleta o vocabulário lexical e os gestos funcionais."""
    return _collect_temporal_labels(
        args, LEXICAL_LABELS + FUNCTIONAL_LABELS, 'palavras e gestos funcionais'
    )


def collect_unknown(args) -> bool:
    """Coleta amostras negativas para a classe de rejeição."""
    return _collect_temporal_labels(args, [UNKNOWN_LABEL], 'fora do vocabulário')


def collect_all(args) -> bool:
    """Coleta o dataset mínimo: alfabeto estático e temporal."""
    print('=== Coleta do dataset mínimo ===')
    return collect_static_data(args) and collect_temporal_data(args)


def dataset_report(args) -> bool:
    """Relatório de cobertura do dataset."""
    from scripts.dataset_report import main as report_main

    sys.argv = ['dataset_report']
    report_main()
    return True


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train_static_model(args) -> bool:
    """Treina o Random Forest estático."""
    print('=== Treino do modelo estático (Random Forest) ===')

    if not _has_static_dataset():
        print('❌ Dataset estático não encontrado. Rode a coleta estática primeiro.')
        return False

    try:
        trainer = LibrasModelTrainer()
        data, labels = trainer.load_dataset()
        trainer.train_model(data, labels)

        model_info = trainer.get_model_info()
        if model_info:
            accuracy = model_info['training_history'].get('accuracy', 'N/A')
            print(f"\n🤖 Tipo: {model_info['model_type']} | Estimadores: {model_info['n_estimators']}")
            if isinstance(accuracy, (int, float)):
                print(f'   Acurácia: {accuracy:.2%}')

        print(f'✅ Modelo estático salvo em {STATIC_MODEL_PATH}')
        return True
    except Exception as error:
        print(f'❌ Erro durante o treino estático: {error}')
        return False


def train_temporal_model(args) -> bool:
    """Treina a LSTM temporal."""
    print('=== Treino do modelo temporal (LSTM) ===')

    if not _has_temporal_dataset():
        print('❌ Dataset temporal não encontrado. Rode a coleta temporal primeiro.')
        return False

    try:
        trainer = LibrasLSTMTrainer()
        data, labels, label_map = trainer.load_sequence_dataset()
        trainer.train_model(data, labels)

        print(f'\n🧠 Sequências: {len(data)} | Classes: {list(label_map.values())}')
        print(f"✅ Modelo temporal salvo em {LSTM_CONFIG['model_path']}")
        return True
    except Exception as error:
        print(f'❌ Erro durante o treino temporal: {error}')
        return False


def train_all_models(args) -> bool:
    """Treina os modelos estático e temporal em sequência."""
    print('=== Treino completo (estático + temporal) ===')

    print('\n📦 Etapa 1/2: modelo estático')
    if not train_static_model(args):
        print('❌ Treino interrompido na etapa estática')
        return False

    print('\n🧠 Etapa 2/2: modelo temporal')
    if not train_temporal_model(args):
        print('❌ Treino interrompido na etapa temporal')
        return False

    print('✅ Treino completo concluído!')
    return True


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

def infer_hybrid(args) -> bool:
    """Inferência híbrida: pipeline temporal com fallback estático."""
    print('=== Inferência híbrida ===')

    if not os.path.exists(STATIC_MODEL_PATH):
        print('❌ Modelo estático não encontrado. Rode `train-static` primeiro.')
        return False
    if not os.path.exists(LSTM_CONFIG['model_path']):
        print('❌ Modelo temporal não encontrado. Rode `train-temporal` primeiro.')
        return False

    try:
        LibrasHybridRealtimeClassifier().start_classification(camera_index=args.camera_index)
        return True
    except Exception as error:
        print(f'❌ Erro durante a inferência híbrida: {error}')
        return False


def infer_static(args) -> bool:
    """Inferência apenas com o modelo estático."""
    print('=== Inferência estática ===')

    if not os.path.exists(STATIC_MODEL_PATH):
        print('❌ Modelo estático não encontrado. Rode `train-static` primeiro.')
        return False

    try:
        LibrasRealtimeClassifier().start_classification(record_video=True)
        return True
    except Exception as error:
        print(f'❌ Erro durante a inferência estática: {error}')
        return False


def infer_temporal(args) -> bool:
    """Inferência apenas com o modelo temporal."""
    print('=== Inferência temporal ===')

    if not os.path.exists(LSTM_CONFIG['model_path']):
        print('❌ Modelo temporal não encontrado. Rode `train-temporal` primeiro.')
        return False

    try:
        LibrasLSTMRealtimeClassifier().start_classification()
        return True
    except Exception as error:
        print(f'❌ Erro durante a inferência temporal: {error}')
        return False


# ---------------------------------------------------------------------------
# Embedded (TFLite INT8 / Pico)
# ---------------------------------------------------------------------------

def embedded_train(args) -> bool:
    """Treina as CNNs quantizadas estática e temporal e exporta o bundle."""
    print('=== Treino embedded (estático + temporal + bundle) ===')

    if not _has_static_dataset() or not _has_temporal_dataset():
        print('❌ Datasets estático e temporal são necessários. Rode a coleta primeiro.')
        return False

    try:
        print('\n📦 Etapa 1/3: CNN estática')
        static_trainer = LibrasEmbeddedCNNTrainer(
            dataset_dir=EMBEDDED_CONFIG['dataset_dir'],
            model_path=EMBEDDED_CONFIG['keras_model_path'],
            tflite_path=EMBEDDED_CONFIG['tflite_model_path'],
            label_map_path=EMBEDDED_CONFIG['label_map_path'],
        )
        data, labels, label_map = static_trainer.load_dataset()
        metrics = static_trainer.train_model(data, labels)
        print(f"   {len(data)} amostras | acurácia {metrics['accuracy']:.2%}")

        print('\n🎞️ Etapa 2/3: CNN temporal')
        temporal_trainer = LibrasEmbeddedTemporalCNNTrainer(
            dataset_dir=EMBEDDED_TEMPORAL_CONFIG['dataset_dir'],
            model_path=EMBEDDED_TEMPORAL_CONFIG['keras_model_path'],
            tflite_path=EMBEDDED_TEMPORAL_CONFIG['tflite_model_path'],
            label_map_path=EMBEDDED_TEMPORAL_CONFIG['label_map_path'],
        )
        data, labels, label_map = temporal_trainer.load_dataset()
        metrics = temporal_trainer.train_model(data, labels)
        print(f"   {len(data)} sequências | acurácia {metrics['accuracy']:.2%}")

        print('\n🧩 Etapa 3/3: bundle')
        return embedded_export(args)
    except Exception as error:
        print(f'❌ Erro durante o treino embedded: {error}')
        return False


def embedded_export(args) -> bool:
    """Empacota modelos quantizados, manifesto e pacote C/C++ do Pico."""
    print('=== Export do bundle embedded ===')

    try:
        manifest = build_embedded_bundle()
        print(f"   Bundle:  {EMBEDDED_BUNDLE_CONFIG['bundle_dir']}")
        print(f"   Estático: {manifest['static']['model_file']}")
        print(f"   Temporal: {manifest['temporal']['model_file']}")
        print(f"   Pico:     {manifest['pico_package']['archive_file']}")
        print('✅ Bundle embedded exportado!')
        return True
    except Exception as error:
        print(f'❌ Erro durante o export embedded: {error}')
        return False


def embedded_check(args) -> bool:
    """Roda o runtime embedded sobre os datasets NPY para validar o bundle."""
    print('=== Verificação do runtime embedded ===')

    try:
        runtime = LibrasEmbeddedRuntime(EMBEDDED_BUNDLE_CONFIG['manifest_path'])
        metrics = runtime.evaluate_datasets()

        print(f"   Amostras estáticas:   {metrics['static_samples']} | acurácia {metrics['static_accuracy']:.2%}")
        print(f"   Sequências temporais: {metrics['temporal_sequences']} | acurácia {metrics['temporal_accuracy']:.2%}")
        print(f"   Acurácia híbrida:     {metrics['hybrid_accuracy']:.2%}")
        print('✅ Verificação embedded concluída!')
        return True
    except Exception as error:
        print(f'❌ Erro durante a verificação embedded: {error}')
        return False


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def run_pipeline(args) -> bool:
    """Coleta → treino → inferência."""
    print('🚀 Pipeline completo do LibrIA')

    steps = [
        ('Coleta do dataset mínimo', collect_all),
        ('Treino dos modelos', train_all_models),
        ('Inferência híbrida', infer_hybrid),
    ]

    for name, step in steps:
        print(f'\n📋 {name}\n' + '-' * 40)
        if not step(args):
            print(f'❌ Pipeline interrompido em: {name}')
            return False

    print('\n🎉 Pipeline completo executado com sucesso!')
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Mapeia comando → NOME da função, resolvido em run_command. Guardar a função
# em si congelaria a referência no import: testes que trocam a implementação
# (e qualquer monkeypatch) seriam ignorados e o comando real rodaria.
COMMANDS = {
    'collect': 'collect_all',
    'collect-static': 'collect_static_data',
    'collect-temporal': 'collect_temporal_data',
    'collect-words': 'collect_words',
    'collect-unknown': 'collect_unknown',
    'report': 'dataset_report',
    'train': 'train_all_models',
    'train-static': 'train_static_model',
    'train-temporal': 'train_temporal_model',
    'infer': 'infer_hybrid',
    'infer-static': 'infer_static',
    'infer-temporal': 'infer_temporal',
    'embedded-train': 'embedded_train',
    'embedded-export': 'embedded_export',
    'embedded-check': 'embedded_check',
    'all': 'run_pipeline',
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='LibrIA — reconhecimento e tradução de Libras',
        epilog=(
            'Os nomes dos comandos são os mesmos dos alvos do Makefile: '
            '`make train-temporal` == `python main.py train-temporal`. '
            'Veja `make help` para o menu completo.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'command',
        choices=sorted(COMMANDS),
        help='comando a executar (ver descrições em `make help`)',
    )

    capture = parser.add_argument_group('metadados da coleta')
    capture.add_argument('--subject', default=COLLECTION_CONFIG['default_subject_id'],
                         help='identificador da pessoa que está sinalizando')
    capture.add_argument('--camera-id', default=COLLECTION_CONFIG['default_camera_id'],
                         help='identificador da câmera (modelo/apelido)')
    capture.add_argument('--environment', default=COLLECTION_CONFIG['default_environment'],
                         help='ambiente da captura (ex.: sala_luz_natural)')
    capture.add_argument('--dominant-hand', default=COLLECTION_CONFIG['default_dominant_hand'],
                         choices=['left', 'right', 'desconhecido'],
                         help='mão dominante da pessoa')

    options = parser.add_argument_group('opções de execução')
    options.add_argument('--camera-index', type=int, default=0, help='índice da webcam')
    options.add_argument('--samples', type=int, default=COLLECTION_CONFIG['static_samples_per_label'],
                         help='amostras estáticas por classe')
    options.add_argument('--sequences', type=int, default=COLLECTION_CONFIG['temporal_samples_per_label'],
                         help='sequências temporais por classe')

    return parser


def run_command(command: str, args=None) -> bool:
    """Executa um comando e devolve sucesso/falha para o exit code."""
    handler_name = COMMANDS.get(command)
    if handler_name is None:
        return False

    handler = globals()[handler_name]

    if args is None:
        args = build_parser().parse_args([command])

    result = handler(args)
    return True if result is None else bool(result)


def main():
    setup_logging()
    args = build_parser().parse_args()
    sys.exit(0 if run_command(args.command, args) else 1)


if __name__ == '__main__':
    main()
