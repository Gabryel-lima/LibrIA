#!/usr/bin/env python3
"""
Script específico para coletar dados das letras J e Z
======================================================

Este script utiliza o coletor de dados modificado para capturar
apenas as letras J e Z que estão faltando no dataset atual.
"""

import sys
import os

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.libras_data_collector import LibrasDataCollector

def main():
    """Função principal para coletar dados de J e Z."""
    print("=== LibrIA - Coletor de J e Z ===")
    print("Este script coleta dados apenas para as letras J e Z.")
    print("\nInstruções:")
    print("- Para a letra J: Faça o sinal de J em Libras (mão em forma de gancho)")
    print("- Para a letra Z: Faça o sinal de Z em Libras (dedo indicador traçando Z)")
    print("- Pressione 'm' para começar a capturar cada mão")
    print("- Pressione 'q' para sair a qualquer momento\n")

    # J = 9, Z = 25 (0-indexed, A=0, B=1, ..., J=9, Z=25)
    specific_classes = [9, 25]

    print(f"Letras a serem coletadas: {[chr(65 + i) for i in specific_classes]}")
    print(f"Classes correspondentes: {specific_classes}")
    print(f"Número de imagens por classe: 300 (150 por mão)\n")

    # Inicializar coletor com classes específicas
    collector = LibrasDataCollector(
        specific_classes=specific_classes,
        dataset_size=150  # 150 imagens por mão = 300 total
    )

    try:
        collector.collect_data()
        print("\n✅ Coleta de dados concluída com sucesso!")
        print("Agora você pode processar os dados executando:")
        print("python src/data_processing/libras_dataset_processor.py")

    except KeyboardInterrupt:
        print("\n\n⚠️  Coleta interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro durante a coleta: {e}")

if __name__ == "__main__":
    main()
