"""
Coletor de Dados para Reconhecimento de Libras
==============================================

Este módulo implementa a coleta de dados via webcam para treinar
o modelo de reconhecimento de linguagem de sinais brasileira (Libras).

Funcionalidades:
- Captura de imagens via webcam
- Organização automática por classe (letra do alfabeto)
- Interface visual para orientação do usuário
- Suporte para coleta de ambas as mãos
"""

import os
import cv2 as cv
from typing import Dict, List

class LibrasDataCollector:
    """Classe para coleta de dados de Libras via webcam."""
    
    def __init__(self, data_dir: str = './data', dataset_size: int = 150):
        """
        Inicializa o coletor de dados.
        
        Args:
            data_dir: Diretório para salvar os dados coletados
            dataset_size: Número de imagens por classe
        """
        self.data_dir = data_dir
        self.dataset_size = dataset_size
        self.number_of_classes = 26
        self.alphabet_dict = {i: chr(65 + i) for i in range(self.number_of_classes) 
                             if chr(65 + i) not in ['J', 'Z']}
        self.hands = ['Right', 'Left']
        
        # Criar diretório de dados se não existir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def collect_data(self):
        """Executa a coleta de dados para todas as classes."""
        cap = cv.VideoCapture(0)
        
        try:
            for class_id in self.alphabet_dict.keys():
                # Pular se a classe já foi coletada
                if os.path.exists(os.path.join(self.data_dir, str(class_id))):
                    print(f'Classe {class_id} ({self.alphabet_dict[class_id]}) já coletada. Pulando...')
                    continue
                
                # Criar diretório para a classe
                os.makedirs(os.path.join(self.data_dir, str(class_id)))
                
                print(f'Coletando dados para classe {class_id} - {self.alphabet_dict[class_id]}')
                
                counter = 0
                for i, hand in enumerate(self.hands):
                    # Aguardar comando para iniciar captura
                    self._wait_for_capture_command(cap, hand)
                    
                    # Capturar imagens
                    counter = self._capture_images(cap, class_id, counter, i)
        
        finally:
            cap.release()
            cv.destroyAllWindows()
    
    def _wait_for_capture_command(self, cap, hand: str):
        """Aguarda o usuário pressionar 'm' para iniciar a captura."""
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            cv.putText(frame, f'Pressione "m" para capturar mão {hand}', 
                      (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow('Coleta de Dados - LibrIA', frame)
            
            if cv.waitKey(25) & 0xFF == ord('m'):
                break
    
    def _capture_images(self, cap, class_id: int, counter: int, hand_index: int) -> int:
        """Captura as imagens para uma mão específica."""
        while counter < self.dataset_size * (hand_index + 1):
            ret, frame = cap.read()
            if not ret:
                continue
                
            cv.imshow('Coleta de Dados - LibrIA', frame)
            cv.waitKey(25)
            
            # Salvar imagem
            img_path = os.path.join(self.data_dir, str(class_id), f'{counter}.jpg')
            cv.imwrite(img_path, frame)
            counter += 1
            
            # Mostrar progresso
            if counter % 10 == 0:
                print(f'  Capturadas {counter} imagens...')
        
        return counter

def main():
    """Função principal para execução do coletor de dados."""
    print("=== LibrIA - Coletor de Dados ===")
    print("Este script irá coletar dados para treinar o modelo de Libras.")
    print("Para cada letra do alfabeto, você deve fazer o sinal correspondente.")
    print("Pressione 'q' a qualquer momento para sair.\n")
    
    collector = LibrasDataCollector()
    collector.collect_data()
    
    print("\nColeta de dados concluída!")

if __name__ == "__main__":
    main()
