#!/usr/bin/env python3
"""
LibrIA - Sistema de Reconhecimento de Libras
============================================

Script principal que integra todas as funcionalidades do projeto:
- Coleta de dados
- Processamento de dataset
- Treinamento de modelo
- Inferência em tempo real

Uso:
    python main.py [comando] [opções]

Comandos disponíveis:
    collect     - Coletar dados via webcam
    process     - Processar dataset coletado
    train       - Treinar modelo
    infer       - Executar inferência em tempo real
    all         - Executar pipeline completo
    help        - Mostrar esta ajuda
"""

import sys
import os
import argparse
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection.libras_data_collector import LibrasDataCollector
from data_processing.libras_dataset_processor import LibrasDatasetProcessor
from model_training.libras_model_trainer import LibrasModelTrainer
from inference.libras_realtime_classifier import LibrasRealtimeClassifier
from config.settings import create_directories, validate_config
from utils.helpers import setup_logging

def collect_data():
    """Executa a coleta de dados."""
    print("=== Iniciando Coleta de Dados ===")
    
    # Validar configurações
    errors = validate_config()
    if errors:
        print("Erros de configuração encontrados:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    # Criar diretórios necessários
    create_directories()
    
    # Executar coleta
    try:
        collector = LibrasDataCollector()
        collector.collect_data()
        print("✅ Coleta de dados concluída com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante a coleta de dados: {e}")
        return False

def process_dataset():
    """Executa o processamento do dataset."""
    print("=== Iniciando Processamento do Dataset ===")
    
    # Verificar se existem dados coletados
    if not os.path.exists('./data'):
        print("❌ Diretório de dados não encontrado. Execute a coleta primeiro.")
        return False
    
    # Executar processamento
    try:
        processor = LibrasDatasetProcessor()
        processor.process_dataset()
        
        # Mostrar informações do dataset
        info = processor.get_dataset_info()
        if info:
            print(f"\n📊 Dataset processado:")
            print(f"   Amostras: {info['num_samples']}")
            print(f"   Features: {info['num_features']}")
            print(f"   Classes: {info['num_classes']}")
        
        print("✅ Processamento do dataset concluído com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        return False

def train_model():
    """Executa o treinamento do modelo."""
    print("=== Iniciando Treinamento do Modelo ===")
    
    # Verificar se existe dataset processado
    if not os.path.exists('./dataset/data.pickle'):
        print("❌ Dataset processado não encontrado. Execute o processamento primeiro.")
        return False
    
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
            print(f"   Acurácia: {model_info['training_history'].get('accuracy', 'N/A'):.2%}")
        
        print("✅ Treinamento do modelo concluído com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        return False

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

def run_pipeline():
    """Executa o pipeline completo."""
    print("🚀 Executando Pipeline Completo do LibrIA")
    print("=" * 50)
    
    steps = [
        ("Coleta de Dados", collect_data),
        ("Processamento do Dataset", process_dataset),
        ("Treinamento do Modelo", train_model),
        ("Inferência em Tempo Real", run_inference)
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

  collect     Coletar dados via webcam para treinar o modelo
  process     Processar dataset coletado (extrair landmarks)
  train       Treinar modelo de machine learning
  infer       Executar reconhecimento em tempo real
  all         Executar pipeline completo (collect → process → train → infer)
  help        Mostrar esta ajuda

EXEMPLOS DE USO:

  # Executar pipeline completo
  python main.py all

  # Apenas coletar dados
  python main.py collect

  # Apenas treinar modelo (se já tiver dados)
  python main.py train

  # Apenas inferência (se já tiver modelo treinado)
  python main.py infer

ESTRUTURA DO PROJETO:

  src/
  ├── data_collection/     # Coleta de dados via webcam
  ├── data_processing/     # Processamento de imagens
  ├── model_training/      # Treinamento de modelos
  ├── inference/          # Inferência em tempo real
  config/                 # Configurações do projeto
  utils/                  # Utilitários e funções auxiliares
  data/                   # Dados coletados (imagens)
  dataset/                # Dataset processado
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
                       choices=['collect', 'process', 'train', 'infer', 'all', 'help'],
                       help='Comando a ser executado')
    
    # Parse argumentos
    args = parser.parse_args()
    
    # Executar comando
    if args.command == 'collect':
        collect_data()
    elif args.command == 'process':
        process_dataset()
    elif args.command == 'train':
        train_model()
    elif args.command == 'infer':
        run_inference()
    elif args.command == 'all':
        run_pipeline()
    else:
        show_help()

if __name__ == "__main__":
    main()
