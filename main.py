# Importações de bibliotecas necessárias
from src.utils.imports import (np, tf, plt, traceback, json)
from src.utils.plots import plot_training_history
import keras
from src.core.libria import Libria
import cv2 as cv

def error_log():
    """Função para registrar o erro em um arquivo log."""
    with open('error_log.txt', 'w') as f:
        f.write('An exception occurred:\n')
        f.write(traceback.format_exc())
    print('An exception occurred. Check error_log.txt for details.')

# Função de aumento de dados
def augment(image, label) -> tuple:
    """Função para realizar o aumento de dados."""
    # Flip horizontal aleatório
    image = tf.image.random_flip_left_right(image)
    # Ajuste aleatório de brilho e contraste
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.1, upper=1.4)
    # Rotação aleatória
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image, label

# Função de treinamento do modelo
def train_model():
    """Função principal para o treinamento do modelo."""
    try:
        libria = Libria(image_shape=(28, 28, 1), landmark_dim=42, num_blocks=[2, 2, 2, 2])
        X_train_img, X_test_img, X_train_landmarks, X_test_landmarks, y_train, y_test = libria.load_data()
        libria.build_model()

        # Resumo do modelo
        libria.model.summary()
        
        # Compilação do modelo
        libria.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Preparar o conjunto de dados de treinamento e validação com duas entradas
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_img, X_train_landmarks), y_train))
        train_dataset = train_dataset.map(lambda x, y: (augment(x[0], y), x[1]), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices(((X_test_img, X_test_landmarks), y_test)).batch(64)

        # Configuração de callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

        # Treinamento do modelo
        history = libria.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20,
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Plot do histórico de treinamento
        plot_training_history(history)
        
        # Avaliação do modelo
        test_loss, test_accuracy = libria.model.evaluate(val_dataset)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        
        # Salvar o modelo e os resultados
        libria.model.save('./model/LibriaCombinedModel.keras')
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'training_loss': history.history['loss'],
            'validation_loss': history.history['val_loss'],
            'training_accuracy': history.history['accuracy'],
            'validation_accuracy': history.history['val_accuracy']
        }
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    except Exception as e:
        error_log()
        print(f'Error: {e}')

# Função de entrada para iniciar o treinamento
def eval_input():
    """Garante o input corretamente"""
    try:
        actions = {
            'train': train_model
        }

        choice = input("\nDigite 'train' para treinar o modelo: ").lower()
        
        if choice in actions:
            actions[choice]()
        else:
            print("Opção inválida. Por favor, tente novamente.")

    except Exception as e:
        error_log()
        print(f"Ocorreu um erro: {e}")

def main():
    eval_input()

if __name__ == '__main__':
    main()
