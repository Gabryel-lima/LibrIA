from src.utils.imports import (
    plt # from matplotlib.pyplot as plt 
)

def plot_training_history(history) -> None:
    """
    Plot the training history including loss and accuracy metrics for training and validation.
    
    Parameters
    ----------
    history : keras.callbacks.History
        Training history containing loss, accuracy, and other metrics.

    Returns
    -------
    None
    """
    plt.figure(figsize=(9, 7))

    # Perda Total
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Acurácia de Classificação
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Perda de Classificação
    if 'class_output_loss' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['class_output_loss'], label='Training Classification Loss')
        plt.plot(history.history['val_class_output_loss'], label='Validation Classification Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Classification Loss')
        plt.legend()
        plt.title('Training and Validation Classification Loss')

    # Perda de Regressão para Landmarks
    if 'landmark_output_loss' in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history['landmark_output_loss'], label='Training Landmark Loss')
        plt.plot(history.history['val_landmark_output_loss'], label='Validation Landmark Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Landmark Loss')
        plt.legend()
        plt.title('Training and Validation Landmark Loss')

    # Ajuste final e salvar a figura
    plt.tight_layout()
    plt.savefig("training_history.png")

def sample_correct_class():
    import numpy as np
    import matplotlib.pyplot as plt
    from src.core.MultiModalGestureNet import Libria  # Certifique-se de que Libria está configurado corretamente

    # Carregar o modelo treinado
    libria = Libria(input_shape=(28, 28, 1), num_blocks=[2, 2, 2, 2])
    X, y = [libria.load_data()[i] for i in (0, 3)]
    libria.ResNet.model.load_weights('./model/LibriaResNet18.keras')  # Substitua pelo caminho correto do arquivo .keras

    # Obter previsões no conjunto de teste
    predictions = libria.ResNet.model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)  # Classe prevista para cada imagem
    true_classes = np.argmax(y, axis=1)  # Classe real para cada imagem

    # Criação de uma amostra de cada classe
    num_classes = len(np.unique(true_classes))
    sample_per_class = {}

    for i, (image, true_class, pred_class) in enumerate(zip(X, true_classes, predicted_classes)):
        # Salva a primeira imagem de cada classe
        if true_class not in sample_per_class:
            sample_per_class[true_class] = (image, true_class, pred_class)
        # Para cada classe, mantém apenas uma amostra
        if len(sample_per_class) == num_classes:
            break

    # Configuração do grid de visualização
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))  # Ajuste o tamanho do grid conforme necessário

    for ax, (true_class, (image, true, pred)) in zip(axes.flat, sample_per_class.items()):
        ax.imshow(image.reshape(28, 28), cmap="gray")
        ax.axis("off")
        # Exibe a classe prevista e a classe real
        ax.set_title(f"Real: {true}, Pred: {pred}", color=("green" if true == pred else "red"))

    plt.suptitle("Sample de Sinais de Mão Classificados", fontsize=16)
    plt.tight_layout()
    plt.show()

# def plot_(history):
#     epochs = [i for i in range(20)]
#     fig , ax = plt.subplots(1,2)
#     train_acc = history.history['accuracy']
#     train_loss = history.history['loss']
#     val_acc = history.history['val_accuracy']
#     val_loss = history.history['val_loss']
#     fig.set_size_inches(16,9)

#     plt.axes(ax[0]).plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
#     plt.axes(ax[0]).plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
#     plt.axes(ax[0]).set_title('Training & Validation Accuracy')
#     plt.axes(ax[0]).legend()
#     plt.axes(ax[0]).set_xlabel("Epochs")
#     plt.axes(ax[0]).set_ylabel("Accuracy")

#     plt.axes(ax[1]).plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
#     plt.axes(ax[1]).plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
#     plt.axes(ax[1]).set_title('Testing Accuracy & Loss')
#     plt.axes(ax[1]).legend()
#     plt.axes(ax[1]).set_xlabel("Epochs")
#     plt.axes(ax[1]).set_ylabel("Loss")
