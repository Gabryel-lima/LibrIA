from src.core.libria_class import Libria
import matplotlib.pyplot as plt
import traceback
import json

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Perda
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


def main():
    try:
        model = Libria(input_shape=(28, 28, 1))
        X_train, X_test, y_train, y_test = model.load_data()

        model.res_net.model.summary()
        model.res_net.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Treinar o modelo
        history = model.res_net.model.fit(x=X_train, y=y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

        # Plotar e salvar o histórico de treinamento
        plot_training_history(history)

        # Avaliação final
        test_loss, test_accuracy = model.res_net.model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

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
