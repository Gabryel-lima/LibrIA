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

⚠️  NOTA: Esta funcionalidade é principalmente para prototipagem.
A coleta real de dados deve usar MediaPipe se disponível.
"""

import os
import cv2 as cv
from typing import Dict, List

class LibrasDataCollector:
    """Classe para coleta de dados de Libras via webcam."""
    
    def __init__(self, data_dir: str = './data', dataset_size: int = 150, specific_classes: List[int] = None):
        """
        Inicializa o coletor de dados.

        Args:
            data_dir: Diretório para salvar os dados coletados
            dataset_size: Número de imagens por classe
            specific_classes: Lista de classes específicas para coletar (opcional)
        """
        self.data_dir = data_dir
        self.dataset_size = dataset_size
        self.number_of_classes = 26
        self.alphabet_dict = {i: chr(65 + i) for i in range(self.number_of_classes)}
        self.hands = ['Right', 'Left']
        self.specific_classes = specific_classes or list(range(self.number_of_classes))

        # Criar diretório de dados se não existir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def collect_data(self):
        """
        Executa a coleta de dados para as classes especificadas.
        
        ⚠️  Requer abertura de webcam. Falha graciosamente se webcam não estiver disponível.
        """
        try:
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Não foi possível abrir a webcam")
                print("   Verifique se a webcam está conectada e disponível")
                return
        except Exception as e:
            print(f"❌ Erro ao acessar webcam: {type(e).__name__}: {e}")
            return

        try:
            for class_id in self.specific_classes:
                if class_id not in self.alphabet_dict:
                    print(f'Classe {class_id} não existe no alfabeto. Pulando...')
                    continue

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
        """
        Aguarda o usuário pressionar 'm' para iniciar a captura.
        
        Requer webcam disponível.
        """
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️  Erro ao ler frame da webcam")
                    continue
                    
                cv.putText(frame, f'Pressione "m" para capturar mão {hand}', 
                          (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                cv.imshow('Coleta de Dados - LibrIA', frame)
                
                if cv.waitKey(25) & 0xFF == ord('m'):
                    break
        except Exception as e:
            print(f"⚠️  Erro ao aguardar comando: {type(e).__name__}: {e}")
    
    def _capture_images(self, cap, class_id: int, counter: int, hand_index: int) -> int:
        """
        Captura as imagens para uma mão específica.
        
        Args:
            cap: VideoCapture object
            class_id: ID da classe
            counter: Contador de imagens
            hand_index: Índice da mão
            
        Returns:
            Novo contador de imagens
        """
        try:
            while counter < self.dataset_size * (hand_index + 1):
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️  Erro ao ler frame")
                    continue
                    
                cv.imshow('Coleta de Dados - LibrIA', frame)
                cv.waitKey(25)
                
                try:
                    # Salvar imagem
                    img_path = os.path.join(self.data_dir, str(class_id), f'{counter}.jpg')
                    cv.imwrite(img_path, frame)
                    counter += 1
                    
                    # Mostrar progresso
                    if counter % 10 == 0:
                        print(f'  Capturadas {counter} imagens...')
                except Exception as e:
                    print(f"⚠️  Erro ao salvar imagem: {type(e).__name__}: {e}")
                    continue
            
            return counter
        except Exception as e:
            print(f"⚠️  Erro durante captura: {type(e).__name__}: {e}")
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

def collect_specific_letters():
    """Coleta dados apenas para as letras J e Z."""
    print("=== LibrIA - Coletor Específico de J e Z ===")
    print("Este script irá coletar dados apenas para as letras J e Z.")
    print("Para cada letra, você deve fazer o sinal correspondente.")
    print("Pressione 'q' a qualquer momento para sair.\n")

    # J = 9, Z = 25 (0-indexed)
    specific_classes = [9, 25]

    print(f"Letras a serem coletadas: {[chr(65 + i) for i in specific_classes]}")
    print(f"Classes correspondentes: {specific_classes}\n")

    collector = LibrasDataCollector(specific_classes=specific_classes)
    collector.collect_data()

    print("\nColeta de dados para J e Z concluída!")

if __name__ == "__main__":
    main()
