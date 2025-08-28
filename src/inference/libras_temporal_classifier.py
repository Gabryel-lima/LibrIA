"""
Classificador Temporal em Tempo Real para Reconhecimento de Libras
================================================================

Este módulo implementa a inferência em tempo real usando CNN temporal
com contexto histórico para reconhecimento de Libras.

Funcionalidades:
- Captura de vídeo em tempo real
- Manutenção de buffer temporal de landmarks
- Classificação usando CNN temporal
- Interface visual aprimorada
- Suavização de predições temporais
"""

import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
from typing import Dict, List, Tuple, Optional
import time
import pickle
import os

class LibrasTemporalClassifier:
    """Classe para classificação temporal em tempo real de Libras."""
    
    def __init__(self, 
                 model_path: str = './model/temporal_cnn_model.h5',
                 config_path: str = './model/temporal_model_config.pickle',
                 min_detection_confidence: float = 0.3,
                 prediction_interval: int = 10):
        """
        Inicializa o classificador temporal.
        
        Args:
            model_path: Caminho para o modelo CNN temporal
            config_path: Caminho para as configurações do modelo
            min_detection_confidence: Confiança mínima para detecção
            prediction_interval: Intervalo entre predições (frames)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.min_detection_confidence = min_detection_confidence
        self.prediction_interval = prediction_interval
        
        # Carregar modelo e configurações
        self.model, self.config = self._load_model_and_config()
        
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
        
        # Buffer temporal para landmarks
        self.sequence_length = self.config.get('sequence_length', 16)
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        
        # Histórico de predições para suavização
        self.prediction_history = deque(maxlen=5)
        
        # Variáveis de controle
        self.counter = 0
        self.last_prediction = None
        self.prediction_confidence = 0.0
        self.is_recording = False
        
        print(f"🧠 Classificador Temporal inicializado:")
        print(f"   - Sequência temporal: {self.sequence_length} frames")
        print(f"   - Classes suportadas: {self.config.get('num_classes', 26)}")
        print(f"   - Modelo: {os.path.basename(model_path)}")
    
    def _load_model_and_config(self) -> Tuple[tf.keras.Model, Dict]:
        """
        Carrega o modelo CNN temporal e suas configurações.
        
        Returns:
            Tupla com modelo e dicionário de configurações
        """
        try:
            # Carregar modelo
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
            print("🔄 Carregando modelo CNN temporal...")
            model = tf.keras.models.load_model(self.model_path)
            
            # Carregar configurações
            config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'rb') as f:
                    config = pickle.load(f)
            else:
                print("⚠️  Configurações não encontradas, usando valores padrão")
                config = {
                    'sequence_length': 16,
                    'num_classes': 26
                }
            
            print("✅ Modelo e configurações carregados com sucesso!")
            return model, config
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def _extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai landmarks de um frame.
        
        Args:
            frame: Frame da webcam
            
        Returns:
            Array com landmarks normalizados ou None se não detectado
        """
        # Converter para RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Processar com MediaPipe
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Pegar apenas a primeira mão detectada
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extrair coordenadas
            x_coords = []
            y_coords = []
            
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
            
            return np.array(normalized_landmarks)
        
        return None
    
    def _add_landmarks_to_buffer(self, landmarks: Optional[np.ndarray]):
        """
        Adiciona landmarks ao buffer temporal.
        
        Args:
            landmarks: Landmarks extraídos ou None
        """
        if landmarks is not None and len(landmarks) == 42:
            self.landmark_buffer.append(landmarks)
        else:
            # Se não detectou mão, adicionar zeros
            self.landmark_buffer.append(np.zeros(42))
    
    def _predict_from_sequence(self) -> Tuple[Optional[str], float]:
        """
        Faz predição baseada na sequência temporal atual.
        
        Returns:
            Tupla com letra prevista e confiança
        """
        if len(self.landmark_buffer) < self.sequence_length:
            return None, 0.0
        
        try:
            # Preparar sequência para predição
            sequence = np.array(list(self.landmark_buffer))
            sequence = sequence.reshape(1, self.sequence_length, 42)
            
            # Fazer predição
            predictions = self.model.predict(sequence, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Converter para letra
            predicted_letter = self.alphabet_dict.get(predicted_class, '?')
            
            return predicted_letter, confidence
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return None, 0.0
    
    def _smooth_predictions(self, prediction: str, confidence: float) -> Tuple[str, float]:
        """
        Suaviza predições usando histórico temporal.
        
        Args:
            prediction: Predição atual
            confidence: Confiança atual
            
        Returns:
            Predição suavizada e confiança média
        """
        # Adicionar ao histórico
        self.prediction_history.append((prediction, confidence))
        
        # Se não temos histórico suficiente, retornar predição atual
        if len(self.prediction_history) < 3:
            return prediction, confidence
        
        # Contar ocorrências de cada letra no histórico
        letter_counts = {}
        total_confidence = 0
        
        for pred, conf in self.prediction_history:
            if pred in letter_counts:
                letter_counts[pred] += conf
            else:
                letter_counts[pred] = conf
            total_confidence += conf
        
        # Encontrar letra mais frequente com maior confiança
        if letter_counts:
            best_letter = max(letter_counts.items(), key=lambda x: x[1])
            avg_confidence = total_confidence / len(self.prediction_history)
            return best_letter[0], avg_confidence
        
        return prediction, confidence
    
    def _draw_info_overlay(self, frame: np.ndarray, 
                          prediction: Optional[str], 
                          confidence: float,
                          buffer_status: int) -> np.ndarray:
        """
        Desenha overlay com informações na tela.
        
        Args:
            frame: Frame original
            prediction: Letra prevista
            confidence: Confiança da predição
            buffer_status: Status do buffer temporal
            
        Returns:
            Frame com overlay
        """
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Fundo para informações
        cv.rectangle(overlay_frame, (10, 10), (w-10, 120), (0, 0, 0), -1)
        cv.rectangle(overlay_frame, (10, 10), (w-10, 120), (0, 255, 0), 2)
        
        # Título
        cv.putText(overlay_frame, 'LibrIA - CNN Temporal', 
                  (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Predição atual
        if prediction:
            pred_text = f'Letra: {prediction} ({confidence*100:.1f}%)'
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
        else:
            pred_text = 'Letra: Detectando...'
            color = (128, 128, 128)
        
        cv.putText(overlay_frame, pred_text, 
                  (20, 65), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Status do buffer temporal
        buffer_text = f'Buffer: {buffer_status}/{self.sequence_length}'
        buffer_color = (0, 255, 0) if buffer_status >= self.sequence_length else (0, 255, 255)
        cv.putText(overlay_frame, buffer_text, 
                  (20, 95), cv.FONT_HERSHEY_SIMPLEX, 0.6, buffer_color, 2)
        
        # Indicador de gravação
        if self.is_recording:
            cv.circle(overlay_frame, (w-30, 30), 10, (0, 0, 255), -1)
            cv.putText(overlay_frame, 'REC', (w-50, 45), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Instruções
        instructions = [
            "Controles:",
            "Q - Sair",
            "R - Gravar",
            "S - Screenshot",
            "C - Limpar buffer"
        ]
        
        start_y = h - 120
        cv.rectangle(overlay_frame, (10, start_y-10), (200, h-10), (0, 0, 0), -1)
        
        for i, instruction in enumerate(instructions):
            cv.putText(overlay_frame, instruction, 
                      (15, start_y + i*20), cv.FONT_HERSHEY_SIMPLEX, 0.4, 
                      (255, 255, 255), 1)
        
        return overlay_frame
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processa um frame para classificação temporal.
        
        Args:
            frame: Frame da webcam
            
        Returns:
            Frame processado com overlay
        """
        # Extrair landmarks do frame atual
        landmarks = self._extract_landmarks_from_frame(frame)
        
        # Adicionar ao buffer temporal
        self._add_landmarks_to_buffer(landmarks)
        
        # Fazer predição se tivermos dados suficientes e for o momento certo
        prediction = None
        confidence = 0.0
        
        if (len(self.landmark_buffer) >= self.sequence_length and 
            self.counter % self.prediction_interval == 0):
            
            raw_prediction, raw_confidence = self._predict_from_sequence()
            if raw_prediction:
                # Suavizar predição
                prediction, confidence = self._smooth_predictions(raw_prediction, raw_confidence)
                self.last_prediction = prediction
                self.prediction_confidence = confidence
        else:
            # Usar última predição válida
            prediction = self.last_prediction
            confidence = self.prediction_confidence
        
        # Desenhar landmarks se detectados
        if landmarks is not None:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        # Adicionar overlay com informações
        processed_frame = self._draw_info_overlay(
            frame, prediction, confidence, len(self.landmark_buffer)
        )
        
        return processed_frame
    
    def run_realtime_classification(self, output_video_path: str = 'output_temporal_cnn.mp4'):
        """
        Executa classificação em tempo real.
        
        Args:
            output_video_path: Caminho para salvar vídeo (opcional)
        """
        print("🎥 Iniciando classificação temporal em tempo real...")
        print("💡 Posicione sua mão na frente da câmera e faça os sinais!")
        
        cap = cv.VideoCapture(0)
        video_writer = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Processar frame
                processed_frame = self._process_frame(frame)
                
                # Mostrar frame
                cv.imshow('LibrIA - CNN Temporal', processed_frame)
                
                # Gravar se necessário
                if self.is_recording and video_writer is not None:
                    video_writer.write(processed_frame)
                
                # Processar teclas
                key = cv.waitKey(25) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._toggle_recording(cap, output_video_path)
                    if self.is_recording and video_writer is None:
                        video_writer = self._setup_video_writer(cap, output_video_path)
                    elif not self.is_recording and video_writer is not None:
                        video_writer.release()
                        video_writer = None
                elif key == ord('s'):
                    self._save_screenshot(processed_frame)
                elif key == ord('c'):
                    self._clear_buffer()
                
                self.counter += 1
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv.destroyAllWindows()
    
    def _toggle_recording(self, cap, output_path: str):
        """Alterna estado de gravação."""
        self.is_recording = not self.is_recording
        status = "iniciada" if self.is_recording else "parada"
        print(f"📹 Gravação {status}")
    
    def _setup_video_writer(self, cap, output_path: str):
        """Configura gravador de vídeo."""
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = 20
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        return cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    def _save_screenshot(self, frame: np.ndarray):
        """Salva screenshot do frame atual."""
        timestamp = int(time.time())
        filename = f'screenshot_temporal_{timestamp}.jpg'
        cv.imwrite(filename, frame)
        print(f"📸 Screenshot salvo: {filename}")
    
    def _clear_buffer(self):
        """Limpa o buffer temporal."""
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        print("🧹 Buffer temporal limpo")

def main():
    """Função principal para execução do classificador temporal."""
    try:
        print("=== LibrIA - Classificação Temporal CNN ===")
        
        # Verificar se modelo existe
        model_path = './model/temporal_cnn_model.h5'
        if not os.path.exists(model_path):
            print("❌ Modelo temporal não encontrado!")
            print("💡 Execute primeiro o treinamento: python -m src.model_training.libras_temporal_cnn_trainer")
            return
        
        # Inicializar classificador
        classifier = LibrasTemporalClassifier()
        
        # Executar classificação em tempo real
        classifier.run_realtime_classification()
        
        print("\n✅ Classificação finalizada!")
        
    except Exception as e:
        print(f"❌ Erro durante a classificação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
