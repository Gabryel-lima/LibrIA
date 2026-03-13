#!/usr/bin/env python3
"""Interface principal do LibrIA para coleta, treino e inferência."""

import sys
import os
import argparse
from pathlib import Path

# Configurar o path do projeto para importações corretas
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importações do projeto
from scripts.collect_dataset import collect_static, collect_temporal
from src.model_training.libras_model_trainer import LibrasModelTrainer
from src.model_training.libras_lstm_trainer import LibrasLSTMTrainer
from src.model_training.libras_embedded_cnn_trainer import LibrasEmbeddedCNNTrainer
from src.model_training.libras_embedded_bundle_exporter import build_embedded_bundle
from src.model_training.libras_embedded_temporal_cnn_trainer import LibrasEmbeddedTemporalCNNTrainer
from src.inference.libras_embedded_runtime import LibrasEmbeddedRuntime
from src.inference.libras_realtime_classifier import LibrasRealtimeClassifier
from src.inference.libras_hybrid_realtime_classifier import LibrasHybridRealtimeClassifier
from src.inference.libras_lstm_realtime_classifier import LibrasLSTMRealtimeClassifier
from config.settings import (
    COLLECTION_CONFIG,
    EMBEDDED_BUNDLE_CONFIG,
    EMBEDDED_CONFIG,
    EMBEDDED_TEMPORAL_CONFIG,
    LEGACY_STATIC_DATASET_PATH,
    LSTM_CONFIG,
    STATIC_LABELS,
    TEMPORAL_DATASET_DIR,
    TEMPORAL_LABELS,
    STATIC_DATASET_DIR,
    create_directories,
    validate_config,
)
from utils.helpers import setup_logging


def _has_static_dataset():
    return os.path.exists(LEGACY_STATIC_DATASET_PATH) or os.path.isdir(STATIC_DATASET_DIR)


def _has_temporal_dataset():
    return os.path.isdir(TEMPORAL_DATASET_DIR) or os.path.isdir('./dataset/sequences')

def _validate_runtime_config():
    """Valida configuração e prepara diretórios do projeto."""
    errors = validate_config()
    if errors:
        print("Erros de configuração encontrados:")
        for error in errors:
            print(f"  - {error}")
        return False

    create_directories()
    return True


def collect_static_data():
    """Executa a coleta estática no formato unificado."""
    print("=== Iniciando Coleta Estática ===")

    if not _validate_runtime_config():
        return False

    try:
        collect_static(
            labels=STATIC_LABELS,
            samples_per_label=COLLECTION_CONFIG['static_samples_per_label'],
            output_dir=STATIC_DATASET_DIR,
            camera_index=0,
        )
        print("✅ Coleta estática concluída com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante a coleta estática: {e}")
        return False

def collect_temporal_data():
    """Executa a coleta temporal no formato unificado."""
    print("=== Iniciando Coleta Temporal ===")

    if not _validate_runtime_config():
        return False

    try:
        collect_temporal(
            labels=TEMPORAL_LABELS,
            num_sequences=COLLECTION_CONFIG['temporal_samples_per_label'],
            seq_length=LSTM_CONFIG['sequence_length'],
            output_dir=TEMPORAL_DATASET_DIR,
            camera_index=0,
        )
        print("✅ Coleta temporal concluída com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante a coleta temporal: {e}")
        return False

def collect_minimal_dataset():
    """Executa a coleta estática e temporal no formato unificado."""
    print("=== Iniciando Coleta do Dataset Mínimo ===")

    if not collect_static_data():
        return False
    if not collect_temporal_data():
        return False

    print("✅ Dataset mínimo coletado com sucesso!")
    return True

def train_model():
    """Executa o treinamento do modelo."""
    print("=== Iniciando Treinamento do Modelo ===")
    
    # Verificar se existe dataset processado
    # Executar treinamento
    try:
        trainer = LibrasModelTrainer()
        data, labels = trainer.load_dataset()
        trainer.train_model(data, labels)
        
        # Mostrar informações do modelo
        model_info = trainer.get_model_info()
        if model_info:
            print(f"\n🤖 Modelo treinado:")
            print(f"   Tipo: {model_info['model_type']}")
            print(f"   Estimadores: {model_info['n_estimators']}")
            accuracy = model_info['training_history'].get('accuracy', 'N/A')
            if isinstance(accuracy, (int, float)):
                print(f"   Acurácia: {accuracy:.2%}")
            else:
                print(f"   Acurácia: {accuracy}")
        
        print("✅ Treinamento do modelo concluído com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        return False

def train_lstm_model():
    """Executa o treinamento do modelo LSTM temporal."""
    print("=== Iniciando Treinamento LSTM ===")

    if not _has_temporal_dataset():
        print("❌ Diretório de sequências não encontrado. Execute a coleta temporal primeiro.")
        return False

    try:
        trainer = LibrasLSTMTrainer()
        data, labels, label_map = trainer.load_sequence_dataset()
        trainer.train_model(data, labels)

        print(f"\n🧠 Modelo temporal treinado:")
        print(f"   Sequências: {len(data)}")
        print(f"   Classes: {list(label_map.values())}")
        print("✅ Treinamento LSTM concluído com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante o treinamento LSTM: {e}")
        return False

def train_embedded_model():
    """Treina uma CNN quantizada estática para deployment embedded a partir de sample_XXX.npy."""
    print("=== Iniciando Treinamento Embedded Tiny CNN ===")

    if not _has_static_dataset():
        print("❌ Dataset estático não encontrado. Execute a coleta estática primeiro.")
        return False

    try:
        trainer = LibrasEmbeddedCNNTrainer(
            dataset_dir=EMBEDDED_CONFIG['dataset_dir'],
            model_path=EMBEDDED_CONFIG['keras_model_path'],
            tflite_path=EMBEDDED_CONFIG['tflite_model_path'],
            label_map_path=EMBEDDED_CONFIG['label_map_path'],
        )
        data, labels, label_map = trainer.load_dataset()
        metrics = trainer.train_model(data, labels)

        print("\n📦 Artefatos embedded gerados:")
        print(f"   Dataset: {len(data)} amostras")
        print(f"   Classes: {list(label_map.values())}")
        print(f"   Input: {EMBEDDED_CONFIG['input_points']}x{EMBEDDED_CONFIG['input_channels']} landmarks")
        print(f"   Acurácia: {metrics['accuracy']:.2%}")
        print(f"   Keras: {EMBEDDED_CONFIG['keras_model_path']}")
        print(f"   TFLite int8: {EMBEDDED_CONFIG['tflite_model_path']}")
        print("✅ Treinamento embedded concluído com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante o treinamento embedded: {e}")
        return False

def train_embedded_temporal_model():
    """Treina uma CNN temporal quantizada para J/Z a partir de seq_XXX.npy."""
    print("=== Iniciando Treinamento Embedded Temporal CNN ===")

    if not _has_temporal_dataset():
        print("❌ Dataset temporal não encontrado. Execute a coleta temporal primeiro.")
        return False

    try:
        trainer = LibrasEmbeddedTemporalCNNTrainer(
            dataset_dir=EMBEDDED_TEMPORAL_CONFIG['dataset_dir'],
            model_path=EMBEDDED_TEMPORAL_CONFIG['keras_model_path'],
            tflite_path=EMBEDDED_TEMPORAL_CONFIG['tflite_model_path'],
            label_map_path=EMBEDDED_TEMPORAL_CONFIG['label_map_path'],
        )
        data, labels, label_map = trainer.load_dataset()
        metrics = trainer.train_model(data, labels)

        print("\n🎞️ Artefatos embedded temporais gerados:")
        print(f"   Sequências: {len(data)}")
        print(f"   Classes: {list(label_map.values())}")
        print(
            f"   Input: {EMBEDDED_TEMPORAL_CONFIG['sequence_length']}x"
            f"{EMBEDDED_TEMPORAL_CONFIG['feature_size']}"
        )
        print(f"   Acurácia: {metrics['accuracy']:.2%}")
        print(f"   Keras: {EMBEDDED_TEMPORAL_CONFIG['keras_model_path']}")
        print(f"   TFLite int8: {EMBEDDED_TEMPORAL_CONFIG['tflite_model_path']}")
        print("✅ Treinamento embedded temporal concluído com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante o treinamento embedded temporal: {e}")
        return False

def train_embedded_models():
    """Treina os modelos embedded estático e temporal em sequência."""
    print("=== Iniciando Treinamento Embedded Completo ===")

    print("\n📦 Etapa 1/2: treinando modelo embedded estático")
    static_success = train_embedded_model()
    if not static_success:
        print("❌ Treinamento embedded completo interrompido na etapa estática")
        return False

    print("\n🎞️ Etapa 2/2: treinando modelo embedded temporal")
    temporal_success = train_embedded_temporal_model()
    if not temporal_success:
        print("❌ Treinamento embedded completo interrompido na etapa temporal")
        return False

    print("\n🧩 Etapa 3/3: exportando bundle embedded combinado")
    export_success = export_embedded_bundle()
    if not export_success:
        print("❌ Treinamento embedded completo interrompido na etapa de export")
        return False

    print("✅ Treinamento embedded completo concluído com sucesso!")
    return True

def export_embedded_bundle():
    """Empacota os dois modelos embedded quantizados com manifesto único."""
    print("=== Exportando Bundle Embedded ===")

    try:
        manifest = build_embedded_bundle()
        print("\n📦 Bundle embedded exportado:")
        print(f"   Diretório: {EMBEDDED_BUNDLE_CONFIG['bundle_dir']}")
        print(f"   Manifesto: {EMBEDDED_BUNDLE_CONFIG['manifest_path']}")
        print(f"   Header runtime: {EMBEDDED_BUNDLE_CONFIG['runtime_header_path']}")
        print(f"   Estático: {manifest['static']['model_file']}")
        print(f"   Temporal: {manifest['temporal']['model_file']}")
        print(f"   Pacote Pico: {manifest['pico_package']['package_dir']}")
        print(f"   Archive Pico: {manifest['pico_package']['archive_file']}")
        print("✅ Export do bundle embedded concluído com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante o export embedded: {e}")
        return False

def run_embedded_inference_check():
    """Valida o bundle embedded rodando inferência nos datasets NPY."""
    print("=== Verificando Runtime Embedded ===")

    try:
        runtime = LibrasEmbeddedRuntime(EMBEDDED_BUNDLE_CONFIG['manifest_path'])
        metrics = runtime.evaluate_datasets()

        print("\n🔎 Verificação do bundle embedded:")
        print(f"   Amostras estáticas: {metrics['static_samples']}")
        print(f"   Acurácia estática: {metrics['static_accuracy']:.2%}")
        print(f"   Sequências temporais: {metrics['temporal_sequences']}")
        print(f"   Acurácia temporal: {metrics['temporal_accuracy']:.2%}")
        print(f"   Acurácia híbrida: {metrics['hybrid_accuracy']:.2%}")
        print(f"   Prioridade temporal: {metrics['temporal_priority_classes']}")
        print("   Contrato de landmarks: MediaPipe continua apenas no host para gerar o dataset; ")
        print("   o runtime embedded consome os mesmos tensores .npy e não importa MediaPipe.")
        print("✅ Verificação embedded concluída com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante a verificação embedded: {e}")
        return False

def train_hybrid_models():
    """Reexecuta o treinamento dos modelos estático e temporal."""
    print("=== Iniciando Treinamento Híbrido ===")

    if not _has_static_dataset():
        print("❌ Dataset estático não encontrado. Execute a coleta estática ou o processamento legado primeiro.")
        return False

    if not _has_temporal_dataset():
        print("❌ Diretório de sequências não encontrado para o modelo temporal. Execute a coleta temporal primeiro.")
        return False

    print("\n📦 Etapa 1/2: retreinando modelo estático")
    static_success = train_model()
    if not static_success:
        print("❌ Treinamento híbrido interrompido na etapa estática")
        return False

    print("\n🧠 Etapa 2/2: retreinando modelo temporal")
    temporal_success = train_lstm_model()
    if not temporal_success:
        print("❌ Treinamento híbrido interrompido na etapa temporal")
        return False

    print("✅ Treinamento híbrido concluído com sucesso!")
    return True

def run_inference():
    """Executa a inferência em tempo real."""
    print("=== Iniciando Inferência em Tempo Real ===")
    
    # Verificar se existe modelo treinado
    if not os.path.exists('./model/model.pickle'):
        print("❌ Modelo treinado não encontrado. Execute o treinamento primeiro.")
        return False
    
    # Executar inferência
    try:
        classifier = LibrasRealtimeClassifier()
        classifier.start_classification(record_video=True)
        print("✅ Inferência concluída!")
        return True
    except Exception as e:
        print(f"❌ Erro durante a inferência: {e}")
        return False

def run_lstm_inference():
    """Executa a inferência temporal em tempo real."""
    print("=== Iniciando Inferência Temporal LSTM ===")

    if not os.path.exists('./model/libras_lstm.keras'):
        print("❌ Modelo LSTM não encontrado. Execute o treinamento LSTM primeiro.")
        return False

    try:
        classifier = LibrasLSTMRealtimeClassifier()
        classifier.start_classification()
        print("✅ Inferência temporal concluída!")
        return True
    except Exception as e:
        print(f"❌ Erro durante a inferência temporal: {e}")
        return False

def run_hybrid_inference():
    """Executa a inferência híbrida em tempo real."""
    print("=== Iniciando Inferência Híbrida ===")

    if not os.path.exists('./model/model.pickle'):
        print("❌ Modelo estático não encontrado. Execute o treinamento primeiro.")
        return False

    if not os.path.exists('./model/libras_lstm.keras'):
        print("❌ Modelo LSTM não encontrado. Execute o treinamento LSTM primeiro.")
        return False

    try:
        classifier = LibrasHybridRealtimeClassifier()
        classifier.start_classification()
        print("✅ Inferência híbrida concluída!")
        return True
    except Exception as e:
        print(f"❌ Erro durante a inferência híbrida: {e}")
        return False

def run_pipeline():
    """Executa o pipeline completo."""
    print("🚀 Executando Pipeline Completo do LibrIA")
    print("=" * 50)
    
    steps = [
        ("Coleta do Dataset Mínimo", collect_minimal_dataset),
        ("Treinamento Híbrido", train_hybrid_models),
        ("Inferência Híbrida", run_hybrid_inference),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        print("-" * 30)
        
        success = step_func()
        if not success:
            print(f"❌ Pipeline interrompido na etapa: {step_name}")
            return False
        
        print(f"✅ {step_name} concluída!")
    
    print("\n🎉 Pipeline completo executado com sucesso!")
    return True

def show_help():
    """Mostra a ajuda do sistema."""
    help_text = """
        LibrIA - Sistema de Reconhecimento de Libras
        ============================================

        Este sistema implementa um pipeline completo para reconhecimento de 
        linguagem de sinais brasileira (Libras) usando visão computacional.

        COMANDOS DISPONÍVEIS:

        collect_static   Coletar dataset estático em dataset/static
        collect_temporal Coletar dataset temporal em dataset/temporal
        collect_minimal  Coletar o dataset mínimo completo
        train       Treinar modelo Random Forest
        train_lstm  Treinar modelo temporal LSTM com dataset/temporal
        train_embedded Treinar CNN estática com sample_XXX.npy para deployment embedded
        train_embedded_temporal Treinar CNN temporal com seq_XXX.npy para J e Z
        train_embedded_all Treinar os dois modelos embedded em sequência
        export_embedded Empacotar os dois modelos quantizados, metadados e pacote C/C++ para Pico
        train_hybrid Retreinar os modelos estático e temporal em sequência
        infer_embedded Verificar o bundle embedded em cima dos datasets NPY
        infer       Executar reconhecimento em tempo real
        infer_lstm  Executar reconhecimento temporal com janela deslizante
        infer_hybrid Executar reconhecimento híbrido com arbitragem
        all         Executar pipeline completo (collect_minimal → train_hybrid → infer_hybrid)
        help        Mostrar esta ajuda

        EXEMPLOS DE USO:

        # Executar pipeline completo
        python main.py all

        # Coleta estática
        python main.py collect_static

        # Coleta temporal
        python main.py collect_temporal

        # Coleta mínima completa
        python main.py collect_minimal

        # Treinar modelo Random Forest
        python main.py train

        # Treinar modelo temporal LSTM
        python main.py train_lstm

        # Treinar CNN embedded estática
        python main.py train_embedded

        # Treinar CNN embedded temporal para J e Z
        python main.py train_embedded_temporal

        # Treinar pipeline embedded completo
        python main.py train_embedded_all

        # Exportar bundle embedded combinado e pacote do Pico
        python main.py export_embedded

        # Verificar runtime embedded com os datasets NPY
        python main.py infer_embedded

        # Retreinar o modo híbrido
        python main.py train_hybrid

        # Executar reconhecimento em tempo real
        python main.py infer

        # Executar reconhecimento temporal
        python main.py infer_lstm

        # Executar reconhecimento híbrido
        python main.py infer_hybrid

        ESTRUTURA DO PROJETO:

        scripts/                # Coleta e calibração de câmera
        src/model_training/     # Treinamento de modelos
        src/inference/          # Inferência em tempo real
        config/                 # Configurações do projeto
        utils/                  # Utilitários e funções auxiliares
        dataset/                # Dataset estático e temporal
        model/                  # Modelos treinados
        output/                 # Saídas (vídeos, screenshots)

        REQUISITOS:

        - Python 3.8+
        - Webcam funcional
        - Dependências listadas em requirements.txt

        Para mais informações, consulte o README.md
        """
    print(help_text)

def main():
    """Função principal."""
    # Configurar logging
    setup_logging()
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(
        description="LibrIA - Sistema de Reconhecimento de Libras",
        add_help=False
    )
    parser.add_argument('command', nargs='?', default='help',
                       choices=['collect_static', 'collect_temporal', 'collect_minimal', 'train', 'train_lstm', 'train_embedded', 'train_embedded_temporal', 'train_embedded_all', 'export_embedded', 'train_hybrid', 'infer', 'infer_lstm', 'infer_hybrid', 'infer_embedded', 'all', 'help'],
                       help='Comando a ser executado')
    
    # Parse argumentos
    args = parser.parse_args()
    
    # Executar comando
    if args.command == 'collect_static':
        collect_static_data()
    elif args.command == 'collect_temporal':
        collect_temporal_data()
    elif args.command == 'collect_minimal':
        collect_minimal_dataset()
    elif args.command == 'train':
        train_model()
    elif args.command == 'train_lstm':
        train_lstm_model()
    elif args.command == 'train_embedded':
        train_embedded_model()
    elif args.command == 'train_embedded_temporal':
        train_embedded_temporal_model()
    elif args.command == 'train_embedded_all':
        train_embedded_models()
    elif args.command == 'export_embedded':
        export_embedded_bundle()
    elif args.command == 'train_hybrid':
        train_hybrid_models()
    elif args.command == 'infer':
        run_inference()
    elif args.command == 'infer_lstm':
        run_lstm_inference()
    elif args.command == 'infer_hybrid':
        run_hybrid_inference()
    elif args.command == 'infer_embedded':
        run_embedded_inference_check()
    elif args.command == 'all':
        run_pipeline()
    else:
        show_help()

if __name__ == "__main__":
    main()
