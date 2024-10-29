from src.utils.imports import (
    np, 
    tf, 
    plt, 
    traceback, 
    json
)
from src.utils.plots import (
    plot_training_history,
    plot_
)

from src.core.libria import Libria

# Função para realizar o aumento de dados
def augment(image, label) -> tuple:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # Rotação aleatória
    return image, label

def main():
    try:
        # Inicializar o modelo com a configuração desejada
        model = Libria(input_shape=(28, 28, 1), num_blocks=[2, 2, 2, 2])
        X_train, X_test, y_train, y_test = model.load_data()

        model.ResNet.model.summary()
        model.ResNet.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Criando o pipeline de dados com tf.data.Dataset e aplicando o aumento de dados
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(64).prefetch(tf.data.AUTOTUNE)

        # Conjunto de validação
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

        # Treinamento do modelo com o tf.data.Dataset
        history = model.ResNet.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=7
        )

        # Plotar e salvar o histórico de treinamento
        plot_training_history(history)
        plot_(history)

        # Avaliação final
        test_loss, test_accuracy = model.ResNet.model.evaluate(val_dataset)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # Realizar predições no conjunto de teste
        predictions = model.ResNet.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)  # Pega o índice da maior probabilidade para cada amostra
        true_classes = np.argmax(y_test, axis=1)  # Converte one-hot para índices

        # Identificar previsões corretas
        correct = np.nonzero(predicted_classes == true_classes)[0]

        # Visualizar algumas imagens corretamente classificadas
        plt.figure(figsize=(10, 10))
        for i, c in enumerate(correct[:6]):
            plt.subplot(3, 2, i + 1)
            plt.imshow(X_test[c].reshape(28, 28), cmap="gray", interpolation='none')
            plt.title(f"Pred: {predicted_classes[c]}, True: {true_classes[c]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("sample_predictions.png")

        # Savar o modelo
        model.ResNet.model.save("./model/LibriaResNet18.keras")

        # Salvar resultados em um arquivo JSON
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "training_loss": history.history['loss'],
            "validation_loss": history.history['val_loss'],
            "training_accuracy": history.history['accuracy'],
            "validation_accuracy": history.history['val_accuracy']
        }

        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        # Captura de exceções e salvamento no arquivo de log
        with open("error_log.txt", "w") as f:
            f.write("An exception occurred:\n")
            f.write(traceback.format_exc())
        print("An exception occurred. Check error_log.txt for details.")

if __name__ == "__main__":
    main()
