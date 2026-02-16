"""
Classificador em Tempo Real para Reconhecimento de Libras
========================================================

Este módulo implementa a inferência em tempo real para reconhecimento
de linguagem de sinais brasileira (Libras) via webcam.

Funcionalidades:
- Captura de vídeo em tempo real
- Detecção de landmarks das mãos
- Classificação de sinais
- Interface visual com feedback
- Gravação de vídeo (opcional)
"""

import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class LibrasRealtimeClassifier:
    """Classe para classificação em tempo real de Libras."""
    
    def __init__(self, model_path: str = './model/model.pickle', min_detection_confidence: float = 0.3, prediction_interval: int = 20):
        """
        Inicializa o classificador em tempo real.
        
        Args:
            model_path: Caminho para o modelo treinado
            min_detection_confidence: Confiança mínima para detecção
            prediction_interval: Intervalo entre predições (frames)
        """
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.prediction_interval = prediction_interval
        
        # Carregar modelo
        self.model = self._load_model()
        
        # Configurar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Inicializar detector de mãos
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Dicionário do alfabeto
        self.alphabet_dict = {i: chr(65 + i) for i in range(26)}
        
        # Variáveis de controle
        self.counter = 0
        self.last_prediction = None
        self.prediction_confidence = 0.0
    
    def _load_model(self):
        """Carrega o modelo treinado."""
        try:
            with open(self.model_path, 'rb') as f:
                model_dict = pickle.load(f)
                return model_dict['model']
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo: {e}")
    
    def start_classification(self, record_video: bool = False, output_path: str = 'output.mp4'):
        """
        Inicia a classificação em tempo real.
        
        Args:
            record_video: Se deve gravar o vídeo
            output_path: Caminho para salvar o vídeo
        """
        print("=== LibrIA - Classificação em Tempo Real ===")
        print("Pressione 'q' para sair")
        print("Pressione 'r' para alternar gravação")
        print("Pressione 's' para capturar screenshot")
        
        # Inicializar captura de vídeo
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Não foi possível abrir a webcam")
        
        # Configurar gravação de vídeo
        video_writer = None
        if record_video:
            video_writer = self._setup_video_recording(cap, output_path)
        
        try:
            while True:
                # Processar frame
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Processar frame para classificação
                processed_frame = self._process_frame(frame)
                
                # Mostrar frame processado
                cv.imshow('LibrIA - Reconhecimento em Tempo Real', processed_frame)
                
                # Gravar frame se necessário
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Processar teclas
                key = cv.waitKey(25) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    video_writer = self._toggle_video_recording(cap, video_writer, output_path)
                elif key == ord('s'):
                    self._save_screenshot(processed_frame)
                
                self.counter += 1
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv.destroyAllWindows()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processa um frame para classificação.
        
        Args:
            frame: Frame da webcam
            
        Returns:
            Frame processado com overlay de informações
        """
        # Converter para RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Processar com MediaPipe
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Desenhar landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Extrair landmarks para classificação
            landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
            
            if landmarks is not None:
                # Fazer predição periodicamente
                if self.counter % self.prediction_interval == 0:
                    self._make_prediction(landmarks)
                
                # Desenhar bounding box e predição
                frame = self._draw_prediction_overlay(frame, results.multi_hand_landmarks[0])
        
        # Adicionar informações na tela
        frame = self._add_info_overlay(frame)
        
        return frame
    
    def _extract_landmarks(self, hand_landmarks) -> Optional[List[float]]:
        """
        Extrai landmarks normalizados da mão.
        
        Args:
            hand_landmarks: Landmarks da mão do MediaPipe
            
        Returns:
            Lista de coordenadas normalizadas
        """
        try:
            x_coords = []
            y_coords = []
            
            # Extrair coordenadas
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
            
            # Normalizar coordenadas
            normalized_landmarks = []
            min_x, min_y = min(x_coords), min(y_coords)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x - min_x
                y = hand_landmarks.landmark[i].y - min_y
                normalized_landmarks.extend([x, y])
            
            return normalized_landmarks
            
        except Exception as e:
            print(f"Erro ao extrair landmarks: {e}")
            return None
    
    def _make_prediction(self, landmarks: List[float]):
        """Faz a predição usando o modelo treinado."""
        try:
            # Fazer predição
            prediction = self.model.predict([np.asarray(landmarks)])
            predicted_class = int(prediction[0])
            
            # Calcular confiança (probabilidade)
            probabilities = self.model.predict_proba([np.asarray(landmarks)])
            confidence = np.max(probabilities[0])
            
            # Atualizar predição atual
            self.last_prediction = predicted_class
            self.prediction_confidence = confidence
            
        except Exception as e:
            print(f"Erro na predição: {e}")
    
    def _draw_prediction_overlay(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        """
        Desenha overlay com a predição no frame.
        
        Args:
            frame: Frame da webcam
            hand_landmarks: Landmarks da mão
            
        Returns:
            Frame com overlay
        """
        if self.last_prediction is None:
            return frame
        
        # Calcular bounding box
        H, W, _ = frame.shape
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        x1 = int(min(x_coords) * W) - 10
        y1 = int(min(y_coords) * H) - 10
        x2 = int(max(x_coords) * W) + 10
        y2 = int(max(y_coords) * H) + 10
        
        # Desenhar retângulo
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Obter letra prevista
        predicted_letter = self.alphabet_dict.get(self.last_prediction, '?')
        
        # Desenhar texto da predição
        text = f"{predicted_letter} ({self.prediction_confidence:.2f})"
        cv.putText(frame, text, (x1, y1 - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)
        
        return frame
    
    def _add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Adiciona informações gerais no frame.
        
        Args:
            frame: Frame da webcam
            
        Returns:
            Frame com informações
        """
        # Adicionar título
        cv.putText(frame, "LibrIA - Reconhecimento de Libras", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Adicionar instruções
        cv.putText(frame, "Pressione 'q' para sair | 'r' para gravar | 's' para screenshot", 
                  (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def _setup_video_recording(self, cap, output_path: str):
        """Configura a gravação de vídeo."""
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # Determinar o codec baseado na extensão do arquivo
        if output_path.lower().endswith('.mp4'):
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
        elif output_path.lower().endswith('.avi'):
            fourcc = cv.VideoWriter_fourcc(*'XVID')
        else:
            # Padrão para MP4
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
        
        return cv.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    
    def _toggle_video_recording(self, cap, video_writer, output_path: str):
        """Alterna a gravação de vídeo."""
        if video_writer is None:
            print("Iniciando gravação...")
            return self._setup_video_recording(cap, output_path)
        else:
            print("Parando gravação...")
            video_writer.release()
            return None
    
    def _save_screenshot(self, frame: np.ndarray):
        """Salva um screenshot do frame atual."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv.imwrite(filename, frame)
        print(f"Screenshot salvo: {filename}")

def main():
    """Função principal para execução do classificador."""
    try:
        classifier = LibrasRealtimeClassifier()
        classifier.start_classification(record_video=True)
        
    except Exception as e:
        print(f"Erro durante a classificação: {e}")

if __name__ == "__main__":
    main()
