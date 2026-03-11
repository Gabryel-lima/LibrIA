"""
Treinador de Modelo para Reconhecimento de Libras
================================================

Este módulo implementa o treinamento do modelo de machine learning
para reconhecimento de linguagem de sinais brasileira (Libras).

Funcionalidades:
- Carregamento e preparação de dados
- Treinamento de modelo Random Forest
- Avaliação de performance
- Salvamento do modelo treinado
"""

import os
import pickle
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Tuple, Any

from config.settings import (
    FEATURE_DIMENSION,
    FEATURE_MODE,
    LEGACY_STATIC_DATASET_PATH,
    MODEL_CONFIG,
    STATIC_DATASET_DIR,
    STATIC_LABELS,
    TRAINING_CONFIG,
)

class LibrasModelTrainer:
    """Classe para treinamento do modelo de Libras."""
    
    def __init__(self, dataset_path: str = LEGACY_STATIC_DATASET_PATH,
                 model_output_dir: str = './model'):
        """
        Inicializa o treinador de modelo.
        
        Args:
            dataset_path: Caminho para o dataset processado
            model_output_dir: Diretório para salvar o modelo treinado
        """
        self.dataset_path = dataset_path
        self.static_dataset_dir = STATIC_DATASET_DIR
        self.model_output_dir = model_output_dir
        self.model = None
        self.training_history = {}

    def _load_from_static_directory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carrega amostras estáticas salvas diretamente em dataset/static."""
        if not os.path.isdir(self.static_dataset_dir):
            raise FileNotFoundError(f"Diretório de dataset estático não encontrado: {self.static_dataset_dir}")

        data = []
        labels = []
        allowed_labels = set(STATIC_LABELS)

        for label in sorted(os.listdir(self.static_dataset_dir)):
            label_dir = os.path.join(self.static_dataset_dir, label)
            if not os.path.isdir(label_dir) or label not in allowed_labels:
                continue

            for filename in sorted(os.listdir(label_dir)):
                if not filename.endswith('.npy') or not filename.startswith('sample_'):
                    continue

                sample_path = os.path.join(label_dir, filename)
                sample = np.load(sample_path)
                flattened = np.asarray(sample, dtype=np.float32).reshape(-1)
                if flattened.shape[0] != FEATURE_DIMENSION:
                    continue

                data.append(flattened)
                labels.append(label)

        if not data:
            raise ValueError(
                'Nenhuma amostra estática válida encontrada em dataset/static com '
                f'{FEATURE_DIMENSION} features.'
            )

        return np.asarray(data), np.asarray(labels)
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega e prepara o dataset para treinamento.
        
        Returns:
            Tupla com dados de treinamento e labels
        """
        print("=== LibrIA - Carregamento de Dataset ===")
        
        data = None
        labels = None

        if os.path.isdir(self.static_dataset_dir):
            print(f"Carregando dataset estático direto de: {self.static_dataset_dir}")
            data, labels = self._load_from_static_directory()
        elif os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'rb') as f:
                data_dict = pickle.load(f)

            print(f"Dataset legado carregado com {len(data_dict['data'])} amostras")
            print(f"Modo de features do treinamento: {FEATURE_MODE} ({FEATURE_DIMENSION} features)")

            filtered_data = []
            filtered_labels = []
            observed_lengths = Counter()

            for x, y in zip(data_dict['data'], data_dict['labels']):
                if isinstance(x, (list, np.ndarray)):
                    observed_lengths[len(x)] += 1
                if isinstance(x, (list, np.ndarray)) and len(x) == FEATURE_DIMENSION:
                    filtered_data.append(x)
                    filtered_labels.append(y)

            if filtered_data:
                data = np.asarray(filtered_data)
                labels = np.asarray(filtered_labels)
            else:
                details = []
                dataset_feature_mode = data_dict.get('feature_mode')
                dataset_num_features = data_dict.get('num_features')

                if dataset_feature_mode is not None:
                    details.append(f'feature_mode do dataset: {dataset_feature_mode}')
                if dataset_num_features is not None:
                    details.append(f'num_features do dataset: {dataset_num_features}')
                if observed_lengths:
                    details.append(f'dimensões encontradas: {dict(observed_lengths)}')

                detail_text = '; '.join(details) if details else 'dataset sem metadados úteis'
                raise ValueError(
                    'Nenhum dado válido encontrado no dataset legado para o FEATURE_MODE atual. '
                    f'Esperado: {FEATURE_DIMENSION} features; {detail_text}.'
                )
        else:
            raise FileNotFoundError(
                'Nenhum dataset estático encontrado. Esperado um destes caminhos: '
                f'{self.dataset_path} ou {self.static_dataset_dir}'
            )
        
        print(f"Dados filtrados: {len(data)} amostras válidas")
        print(f"Número de features: {data.shape[1]}")
        print(f"Número de classes: {len(np.unique(labels))}")
        
        return data, labels
    
    def train_model(self, data: np.ndarray, labels: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42):
        """
        Treina o modelo de machine learning.
        
        Args:
            data: Dados de treinamento
            labels: Labels correspondentes
            test_size: Proporção de dados para teste
            random_state: Seed para reprodutibilidade
        """
        print("\n=== LibrIA - Treinamento do Modelo ===")
        
        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, shuffle=True, 
            stratify=labels, random_state=random_state
        )
        
        print(f"Dados de treino: {X_train.shape[0]} amostras")
        print(f"Dados de teste: {X_test.shape[0]} amostras")
        
        # Inicializar modelo
        self.model = RandomForestClassifier(
            n_estimators=MODEL_CONFIG['n_estimators'],
            random_state=random_state,
            n_jobs=MODEL_CONFIG['n_jobs']
        )

        self.training_history = {
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'num_features': X_train.shape[1],
            'num_classes': len(np.unique(labels)),
            'classes': sorted(np.unique(labels).tolist()),
            'feature_mode': FEATURE_MODE,
        }
        
        # Treinar modelo
        print("Treinando modelo Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Avaliar modelo
        self._evaluate_model(X_test, y_test)
        
        # Salvar modelo
        self._save_model()
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Avalia o modelo treinado.
        
        Args:
            X_test: Dados de teste
            y_test: Labels de teste
        """
        print("\n=== Avaliação do Modelo ===")
        
        # Predições
        y_pred = self.model.predict(X_test)
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Validação cruzada
        cv_folds = min(TRAINING_CONFIG['cv_folds'], len(y_test))
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=cv_folds)
        print(f"Acurácia CV ({cv_folds}-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Relatório detalhado
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nMatriz de Confusão ({cm.shape[0]}x{cm.shape[1]}):")
        print(cm)
        
        # Salvar métricas
        self.training_history.update({
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
    
    def _save_model(self):
        """Salva o modelo treinado em disco."""
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)
        
        # Salvar modelo
        model_path = os.path.join(self.model_output_dir, 'model.pickle')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'training_history': self.training_history,
                'feature_mode': FEATURE_MODE,
                'num_features': self.training_history.get('num_features', FEATURE_DIMENSION),
            }, f)
        
        print(f"\nModelo salvo em: {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo treinado.
        
        Returns:
            Dicionário com informações do modelo
        """
        if self.model is None:
            return None
        
        return {
            'model_type': type(self.model).__name__,
            'feature_importance': self.model.feature_importances_.tolist(),
            'n_estimators': self.model.n_estimators,
            'feature_mode': self.training_history.get('feature_mode', FEATURE_MODE),
            'training_history': self.training_history
        }

def main():
    """Função principal para execução do treinamento."""
    try:
        # Inicializar treinador
        trainer = LibrasModelTrainer()
        
        # Carregar dataset
        data, labels = trainer.load_dataset()
        
        # Treinar modelo
        trainer.train_model(data, labels)
        
        # Mostrar informações do modelo
        model_info = trainer.get_model_info()
        if model_info:
            print("\n=== Informações do Modelo ===")
            print(f"Tipo: {model_info['model_type']}")
            print(f"Número de estimadores: {model_info['n_estimators']}")
            print(f"Importância média das features: {np.mean(model_info['feature_importance']):.4f}")
        
        print("\nTreinamento concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")

if __name__ == "__main__":
    main()
