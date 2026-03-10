"""
Classificador em Tempo Real para Reconhecimento de Libras
========================================================

Este módulo implementa a inferência em tempo real para reconhecimento
de linguagem de sinais brasileira (Libras) via webcam.

Funcionalidades:
- Captura de vídeo em tempo real
- Detecção de landmarks das mãos (com MediaPipe)
- Classificação de sinais
- Interface visual com feedback
- Gravação de vídeo (opcional)

⚠️  NOTA: MediaPipe requer suporte AVX na CPU.
Se não disponível, apenas modo de teste com dados sintéticos é permitido.
"""

import cv2 as cv
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from config.settings import CAMERA_CONFIG, FEATURE_MODE as DEFAULT_FEATURE_MODE
from utils.helpers import (
    extract_landmarks_by_mode,
    get_feature_dimension,
    infer_feature_mode_from_dimension,
    load_camera_calibration,
    preprocess_frame,
)

# Tentar importar MediaPipe (requer suporte AVX)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️  MediaPipe não disponível: {type(e).__name__}")
    print("   → Modo de inferência com webcam desabilitado")
    print("   → Apenas modo de teste com dados sintéticos disponível")
    mp = None

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
        self.model_metadata = {}
        
        # Carregar modelo
        self.model = self._load_model()
        self.feature_mode = self._resolve_feature_mode()
        self.feature_dimension = get_feature_dimension(self.feature_mode)
        self.camera_calibration = self._load_camera_calibration()
        
        # Configurar MediaPipe (se disponível)
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                # Inicializar detector de mãos
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"⚠️  Erro ao inicializar MediaPipe: {type(e).__name__}: {e}")
                self.hands = None
        else:
            self.hands = None
        
        # Dicionário do alfabeto
        self.alphabet_dict = {i: chr(65 + i) for i in range(26)}
        
        # Variáveis de controle
        self.counter = 0
        self.last_prediction = None
        self.prediction_confidence = 0.0
    
    def _load_model(self):
        """
        Carrega o modelo treinado.
        
        Retorna None se o modelo não existir (modo gracioso).
        """
        try:
            with open(self.model_path, 'rb') as f:
                model_dict = pickle.load(f)
                self.model_metadata = {
                    key: value for key, value in model_dict.items() if key != 'model'
                }
                print(f"✓ Modelo carregado com sucesso: {self.model_path}")
                return model_dict.get('model', None)
        except FileNotFoundError:
            print(f"⚠️  Modelo não encontrado: {self.model_path}")
            print("   → Será usado modo de teste/demo sem modelo real")
            return None
        except Exception as e:
            print(f"⚠️  Erro ao carregar modelo: {type(e).__name__}: {e}")
            print("   → Será usado modo de teste/demo")
            return None

    def _resolve_feature_mode(self) -> str:
        """Resolve o modo de features com base na configuração e nos metadados do modelo."""
        metadata_mode = self.model_metadata.get('feature_mode')
        if metadata_mode:
            return metadata_mode

        if self.model is not None and hasattr(self.model, 'n_features_in_'):
            try:
                return infer_feature_mode_from_dimension(int(self.model.n_features_in_))
            except ValueError:
                pass

        return DEFAULT_FEATURE_MODE

    def _load_camera_calibration(self) -> Optional[Dict[str, np.ndarray]]:
        """Carrega parâmetros de calibração, se estiverem habilitados."""
        if not CAMERA_CONFIG['enabled']:
            return None

        calibration = load_camera_calibration(
            CAMERA_CONFIG['camera_matrix_path'],
            CAMERA_CONFIG['dist_coeffs_path'],
        )
        if calibration:
            print("✓ Calibração de câmera carregada")
        return calibration
    
    def start_classification(self, record_video: bool = False, output_path: str = 'output.mp4'):
        """
        Inicia a classificação em tempo real.
        
        Args:
            record_video: Se deve gravar o vídeo
            output_path: Caminho para salvar o vídeo
        """
        if not MEDIAPIPE_AVAILABLE:
            print("❌ MediaPipe não disponível para inferência em tempo real")
            print("   Motivo: CPU não suporta instruções AVX")
            self._run_test_mode()
            return
        
        if self.hands is None:
            print("❌ Não foi possível inicializar detector de mãos")
            self._run_test_mode()
            return
        
        print("=== LibrIA - Classificação em Tempo Real ===")
        print("Pressione 'q' para sair")
        print("Pressione 'r' para alternar gravação")
        print("Pressione 's' para capturar screenshot")
        
        # Inicializar captura de vídeo
        try:
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Não foi possível abrir a webcam")
                self._run_test_mode()
                return
        except Exception as e:
            print(f"❌ Erro ao acessar webcam: {type(e).__name__}: {e}")
            self._run_test_mode()
            return
        
        # Configurar gravação de vídeo
        video_writer = None
        if record_video:
            try:
                video_writer = self._setup_video_recording(cap, output_path)
            except Exception as e:
                print(f"⚠️  Erro ao configurar gravação: {e}")
        
        try:
            while True:
                # Processar frame
                ret, frame = cap.read()
                if not ret:
                    continue
                
                try:
                    # Processar frame para classificação
                    processed_frame = self._process_frame(frame)
                    
                    # Mostrar frame processado
                    cv.imshow('LibrIA - Reconhecimento em Tempo Real', processed_frame)
                    
                    # Gravar frame se necessário
                    if video_writer:
                        try:
                            video_writer.write(processed_frame)
                        except Exception as e:
                            print(f"⚠️  Erro ao gravar frame: {e}")
                    
                    # Processar teclas
                    key = cv.waitKey(25) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        try:
                            video_writer = self._toggle_video_recording(cap, video_writer, output_path)
                        except Exception as e:
                            print(f"⚠️  Erro ao alternar gravação: {e}")
                    elif key == ord('s'):
                        try:
                            self._save_screenshot(processed_frame)
                        except Exception as e:
                            print(f"⚠️  Erro ao salvar screenshot: {e}")
                    
                    self.counter += 1
                
                except Exception as e:
                    print(f"⚠️  Erro ao processar frame: {type(e).__name__}: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ Erro durante classificação: {type(e).__name__}: {e}")
        
        finally:
            cap.release()
            if video_writer:
                try:
                    video_writer.release()
                except:
                    pass
            cv.destroyAllWindows()
    
    def _run_test_mode(self):
        """
        Modo de teste - simula inferência com dados aleatórios.
        Útil para testar o pipeline sem MediaPipe.
        """
        print("\n" + "="*60)
        print("🧪 MODO DE TESTE - Inferência com Dados Sintéticos")
        print("="*60)
        print("⚠️  Sem MediaPipe: usando landmarks aleatórios")
        print("   Pressione 'q' para sair\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n--- Iteração {iteration} ---")
                
                synthetic_landmarks = np.random.rand(self.feature_dimension).astype(np.float32)
                
                # Fazer predição
                if self.model is not None:
                    try:
                        prediction = self.model.predict([synthetic_landmarks])[0]
                        confidence = self.model.predict_proba([synthetic_landmarks])[0].max()
                        
                        label = self.alphabet_dict.get(prediction, "?")
                        print(f"✓ Predição: {label} (confiança: {confidence:.2%})")
                    except Exception as e:
                        print(f"⚠️  Erro na predição: {type(e).__name__}: {e}")
                else:
                    print("ℹ️  Sem modelo - apenas teste de estrutura")
                    print(f"   Landmarks sintéticos gerados: {synthetic_landmarks[:5]}...")
                
                # Simular delay
                time.sleep(2)
                
                # Permitir sair com 'q'
                try:
                    key = input("  Pressione Enter para próxima iteração (q + Enter para sair): ").strip().lower()
                    if key == 'q':
                        print("✓ Teste interrompido pelo usuário")
                        break
                except EOFError:
                    # Se não puder ler entrada (ex: em modo não-interativo)
                    print("ℹ️  Modo não-interativo detectado, encerrando teste")
                    break
        
        except KeyboardInterrupt:
            print("\n✓ Teste interrompido pelo usuário")
        except Exception as e:
            print(f"\n❌ Erro no modo de teste: {type(e).__name__}: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processa um frame para classificação.
        
        Args:
            frame: Frame da webcam
            
        Returns:
            Frame processado com overlay de informações
        """
        try:
            frame = preprocess_frame(frame, self.camera_calibration)

            # Converter para RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Processar com MediaPipe
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                # Desenhar landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    try:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    except Exception as e:
                        print(f"⚠️  Erro ao desenhar landmarks: {e}")
                
                try:
                    # Extrair landmarks para classificação
                    landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
                    
                    if landmarks is not None:
                        # Fazer predição periodicamente
                        if self.counter % self.prediction_interval == 0:
                            try:
                                self._make_prediction(landmarks)
                            except Exception as e:
                                print(f"⚠️  Erro na predição: {e}")
                        
                        # Desenhar bounding box e predição
                        frame = self._draw_prediction_overlay(frame, results.multi_hand_landmarks[0])
                except Exception as e:
                    print(f"⚠️  Erro ao extrair/processar landmarks: {e}")
            
            # Adicionar informações na tela
            frame = self._add_info_overlay(frame)
            
            return frame
        
        except Exception as e:
            print(f"❌ Erro ao processar frame: {type(e).__name__}: {e}")
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
            features = extract_landmarks_by_mode(hand_landmarks.landmark, self.feature_mode)
            return features.tolist()
            
        except Exception as e:
            print(f"⚠️  Erro ao extrair landmarks: {type(e).__name__}: {e}")
            return None
    
    def _make_prediction(self, landmarks: List[float]):
        """
        Faz a predição usando o modelo treinado.
        
        Requer um modelo scikit-learn com métodos predict() e predict_proba().
        """
        try:
            if self.model is None:
                return
            
            # Fazer predição
            prediction = self.model.predict([np.asarray(landmarks)])
            predicted_class = int(prediction[0])
            
            # Calcular confiança (probabilidade)
            try:
                probabilities = self.model.predict_proba([np.asarray(landmarks)])
                confidence = np.max(probabilities[0])
            except AttributeError:
                # Se o modelo não tem predict_proba, usar um valor padrão
                confidence = 0.0
            
            # Atualizar predição atual
            self.last_prediction = predicted_class
            self.prediction_confidence = confidence
            
        except Exception as e:
            print(f"⚠️  Erro na predição: {type(e).__name__}: {e}")
    
    def _draw_prediction_overlay(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        """
        Desenha overlay com a predição no frame.
        
        Args:
            frame: Frame da webcam
            hand_landmarks: Landmarks da mão
            
        Returns:
            Frame com overlay
        """
        try:
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
        
        except Exception as e:
            print(f"⚠️  Erro ao desenhar predição: {type(e).__name__}: {e}")
            return frame
    
    def _add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Adiciona informações gerais no frame.
        
        Args:
            frame: Frame da webcam
            
        Returns:
            Frame com informações
        """
        try:
            # Adicionar título
            cv.putText(frame, "LibrIA - Reconhecimento de Libras", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Adicionar instruções
            cv.putText(frame, "Pressione 'q' para sair | 'r' para gravar | 's' para screenshot", 
                      (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            return frame
        
        except Exception as e:
            print(f"⚠️  Erro ao adicionar overlay: {type(e).__name__}: {e}")
            return frame
    
    def _setup_video_recording(self, cap, output_path: str):
        """
        Configura a gravação de vídeo.
        
        Args:
            cap: VideoCapture object
            output_path: Caminho para salvar o vídeo
            
        Returns:
            VideoWriter object ou None se falhar
        """
        try:
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
            
            writer = cv.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
            print(f"✓ Gravação iniciada: {output_path}")
            return writer
        
        except Exception as e:
            print(f"⚠️  Erro ao configurar gravação: {type(e).__name__}: {e}")
            return None
    
    def _toggle_video_recording(self, cap, video_writer, output_path: str):
        """
        Alterna a gravação de vídeo.
        
        Args:
            cap: VideoCapture object
            video_writer: VideoWriter atual ou None
            output_path: Caminho para salvar novo vídeo
            
        Returns:
            Novo VideoWriter ou None
        """
        try:
            if video_writer is None:
                print("▶️  Iniciando gravação...")
                return self._setup_video_recording(cap, output_path)
            else:
                print("⏹️  Parando gravação...")
                video_writer.release()
                return None
        except Exception as e:
            print(f"⚠️  Erro ao alternar gravação: {type(e).__name__}: {e}")
            return video_writer
    
    def _save_screenshot(self, frame: np.ndarray):
        """
        Salva um screenshot do frame atual.
        
        Args:
            frame: Frame a ser salvo
        """
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv.imwrite(filename, frame)
            print(f"📸 Screenshot salvo: {filename}")
        except Exception as e:
            print(f"⚠️  Erro ao salvar screenshot: {type(e).__name__}: {e}")

def main():
    """Função principal para execução do classificador."""
    try:
        print("="*60)
        print("LibrIA - Classificador em Tempo Real para Libras")
        print("="*60 + "\n")
        
        classifier = LibrasRealtimeClassifier()
        classifier.start_classification(record_video=False)
        
        print("\n✓ Classificação encerrada com sucesso")
        
    except Exception as e:
        print(f"❌ Erro durante a classificação: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
