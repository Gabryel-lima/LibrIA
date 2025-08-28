"""
Treinador de CNN Temporal para Reconhecimento de Libras
======================================================

Este módulo implementa uma CNN com contexto temporal usando LSTM/GRU
para reconhecimento de linguagem de sinais brasileira (Libras).

Funcionalidades:
- CNN para extração de features espaciais de frames
- LSTM/GRU para modelar contexto temporal
- Processamento de sequências de landmarks
- Treinamento end-to-end com dados temporais
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Dense, Flatten,
    LSTM, GRU, TimeDistributed, Input, 
    BatchNormalization, GlobalAveragePooling2D,
    Reshape, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Any, List
import cv2 as cv

class LibrasTemporalCNNTrainer:
    """Classe para treinamento da CNN temporal de Libras."""
    
    def __init__(self, 
                 dataset_path: str = './dataset/data.pickle',
                 data_dir: str = './data',
                 model_output_dir: str = './model',
                 sequence_length: int = 16,
                 image_size: int = 64):
        """
        Inicializa o treinador da CNN temporal.
        
        Args:
            dataset_path: Caminho para o dataset de landmarks
            data_dir: Diretório com as imagens originais
            model_output_dir: Diretório para salvar o modelo
            sequence_length: Comprimento das sequências temporais
            image_size: Tamanho das imagens para a CNN
        """
        self.dataset_path = dataset_path
        self.data_dir = data_dir
        self.model_output_dir = model_output_dir
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.num_classes = 26  # A-Z
        
        self.model = None
        self.training_history = {}
        self.label_encoder = LabelEncoder()
    
    def create_temporal_cnn_model(self) -> Model:
        """
        Cria a arquitetura da CNN temporal.
        
        Returns:
            Modelo CNN temporal compilado
        """
        print("=== Criando Arquitetura CNN Temporal ===")
        
        # Branch 1: CNN para features visuais (sequência de frames)
        image_input = Input(shape=(self.sequence_length, self.image_size, self.image_size, 3), 
                           name='image_sequence')
        
        # TimeDistributed CNN para processar cada frame
        cnn_features = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same'))(image_input)
        cnn_features = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same'))(cnn_features)
        cnn_features = TimeDistributed(MaxPooling2D(padding='same'))(cnn_features)
        cnn_features = TimeDistributed(Dropout(0.2))(cnn_features)
        
        # Block 2
        cnn_features = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same'))(cnn_features)
        cnn_features = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same'))(cnn_features)
        cnn_features = TimeDistributed(MaxPooling2D(padding='same'))(cnn_features)
        cnn_features = TimeDistributed(Dropout(0.3))(cnn_features)
        
        # Block 3
        cnn_features = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same'))(cnn_features)
        cnn_features = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same'))(cnn_features)
        cnn_features = TimeDistributed(GlobalAveragePooling2D())(cnn_features)  # Em vez de Flatten
        cnn_features = TimeDistributed(Dropout(0.4))(cnn_features)
        
        # Branch 2: LSTM para landmarks temporais
        landmark_input = Input(shape=(self.sequence_length, 42), name='landmark_sequence')  # 21 landmarks × 2 coords
        
        # Processamento dos landmarks
        landmark_features = Dense(64, activation='relu')(landmark_input)
        landmark_features = Dropout(0.2)(landmark_features)
        landmark_features = Dense(32, activation='relu')(landmark_features)
        
        # Combinar features visuais e landmarks
        combined_features = Concatenate(axis=-1)([cnn_features, landmark_features])
        
        # LSTM para contexto temporal
        temporal_features = LSTM(256, return_sequences=True, dropout=0.3)(combined_features)
        temporal_features = LSTM(128, dropout=0.3)(temporal_features)
        
        # Camadas finais de classificação
        dense_features = Dense(512, activation='relu')(temporal_features)
        dense_features = Dropout(0.5)(dense_features)
        dense_features = Dense(256, activation='relu')(dense_features)
        dense_features = Dropout(0.3)(dense_features)
        
        # Saída
        output = Dense(self.num_classes, activation='softmax', name='classification')(dense_features)
        
        # Criar modelo
        model = Model(inputs=[image_input, landmark_input], outputs=output)
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Arquitetura criada com sucesso!")
        model.summary()
        
        return model
    
    def create_landmark_only_model(self) -> Model:
        """
        Cria um modelo simplificado apenas com landmarks para retrocompatibilidade.
        
        Returns:
            Modelo baseado apenas em landmarks
        """
        print("=== Criando Modelo com Landmarks Apenas ===")
        
        # Input para sequência de landmarks
        landmark_input = Input(shape=(self.sequence_length, 42), name='landmark_sequence')
        
        # Camadas densas para processar landmarks
        features = TimeDistributed(Dense(128, activation='relu'))(landmark_input)
        features = TimeDistributed(Dropout(0.2))(features)
        features = TimeDistributed(Dense(64, activation='relu'))(features)
        features = TimeDistributed(Dropout(0.3))(features)
        
        # LSTM para contexto temporal
        temporal_features = LSTM(256, return_sequences=True, dropout=0.3)(features)
        temporal_features = LSTM(128, dropout=0.3)(temporal_features)
        
        # Camadas finais
        dense_features = Dense(512, activation='relu')(temporal_features)
        dense_features = Dropout(0.5)(dense_features)
        dense_features = Dense(256, activation='relu')(dense_features)
        dense_features = Dropout(0.3)(dense_features)
        
        # Saída
        output = Dense(self.num_classes, activation='softmax')(dense_features)
        
        # Criar e compilar modelo
        model = Model(inputs=landmark_input, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Modelo simplificado criado!")
        model.summary()
        
        return model
    
    def prepare_temporal_dataset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepara dataset temporal a partir dos dados existentes.
        
        Returns:
            Dicionário com dados temporais e labels
        """
        print("=== Preparando Dataset Temporal ===")
        
        # Carregar dataset de landmarks existente
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset não encontrado: {self.dataset_path}")
        
        with open(self.dataset_path, 'rb') as f:
            landmark_data = pickle.load(f)
        
        print(f"Dataset de landmarks carregado: {len(landmark_data['data'])} amostras")
        
        # Organizar dados por classe
        class_data = {}
        for landmarks, label in zip(landmark_data['data'], landmark_data['labels']):
            if isinstance(landmarks, (list, np.ndarray)) and len(landmarks) == 42:
                if label not in class_data:
                    class_data[label] = []
                class_data[label].append(np.array(landmarks))
        
        print(f"Dados organizados em {len(class_data)} classes")
        
        # Criar sequências temporais
        temporal_landmarks = []
        temporal_labels = []
        
        for class_id, landmarks_list in class_data.items():
            if len(landmarks_list) < self.sequence_length:
                print(f"Classe {class_id}: poucos dados ({len(landmarks_list)}), pulando...")
                continue
            
            # Criar sequências deslizantes
            for i in range(len(landmarks_list) - self.sequence_length + 1):
                sequence = landmarks_list[i:i + self.sequence_length]
                temporal_landmarks.append(np.array(sequence))
                temporal_labels.append(int(class_id))
        
        temporal_landmarks = np.array(temporal_landmarks)
        temporal_labels = np.array(temporal_labels)
        
        print(f"Dataset temporal criado:")
        print(f"- Sequências de landmarks: {temporal_landmarks.shape}")
        print(f"- Labels: {temporal_labels.shape}")
        print(f"- Classes únicas: {len(np.unique(temporal_labels))}")
        
        return {
            'landmark_sequences': temporal_landmarks,
            'labels': temporal_labels
        }
    
    def train_model(self, model_type: str = 'landmark_only'):
        """
        Treina o modelo selecionado.
        
        Args:
            model_type: Tipo do modelo ('landmark_only' ou 'full_temporal')
        """
        print(f"\n=== Treinamento do Modelo ({model_type}) ===")
        
        # Preparar dados
        dataset = self.prepare_temporal_dataset()
        
        # Dividir dados
        X_landmarks = dataset['landmark_sequences']
        y = dataset['labels']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_landmarks, y, test_size=0.2, shuffle=True, 
            stratify=y, random_state=42
        )
        
        print(f"Dados de treino: {X_train.shape[0]} sequências")
        print(f"Dados de teste: {X_test.shape[0]} sequências")
        
        # Criar modelo
        if model_type == 'landmark_only':
            self.model = self.create_landmark_only_model()
            train_data = X_train
            test_data = X_test
        else:
            # Para implementação futura com imagens
            self.model = self.create_landmark_only_model()
            train_data = X_train
            test_data = X_test
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.model_output_dir, 'best_temporal_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Criar diretório se não existir
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)
        
        # Treinar modelo
        print("Iniciando treinamento...")
        history = self.model.fit(
            train_data, y_train,
            validation_data=(test_data, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Avaliar modelo
        self._evaluate_model(test_data, y_test)
        
        # Salvar modelo e histórico
        self._save_model()
        self.training_history = history.history
        
        print("Treinamento concluído!")
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Avalia o modelo treinado.
        
        Args:
            X_test: Dados de teste
            y_test: Labels de teste
        """
        print("\n=== Avaliação do Modelo ===")
        
        # Predições
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia no teste: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Relatório de classificação
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        # Salvar métricas
        self.training_history.update({
            'test_accuracy': accuracy,
            'num_classes': len(np.unique(y_test))
        })
    
    def _save_model(self):
        """Salva o modelo treinado."""
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)
        
        # Salvar modelo Keras
        model_path = os.path.join(self.model_output_dir, 'temporal_cnn_model.h5')
        self.model.save(model_path)
        
        # Salvar configurações
        config_path = os.path.join(self.model_output_dir, 'temporal_model_config.pickle')
        with open(config_path, 'wb') as f:
            pickle.dump({
                'sequence_length': self.sequence_length,
                'image_size': self.image_size,
                'num_classes': self.num_classes,
                'training_history': self.training_history
            }, f)
        
        print(f"Modelo salvo em: {model_path}")
        print(f"Configurações salvas em: {config_path}")

def main():
    """Função principal para treinamento."""
    try:
        print("=== LibrIA - Treinamento CNN Temporal ===")
        
        # Inicializar treinador
        trainer = LibrasTemporalCNNTrainer(
            sequence_length=16,  # 16 frames de contexto
            image_size=64        # Imagens 64x64 para eficiência
        )
        
        # Treinar modelo (começar apenas com landmarks)
        trainer.train_model(model_type='landmark_only')
        
        print("\n🎉 Treinamento da CNN Temporal concluído com sucesso!")
        print("O modelo agora possui contexto temporal para melhor reconhecimento!")
        
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
