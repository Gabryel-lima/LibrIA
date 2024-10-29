from src.utils.imports import (
    plt # from matplotlib.pyplot as plt 
)

def plot_training_history(history) -> None:
    """
    Um histograma simples de linha
    
    Parameters
    ----------
    dtype : history = numpy.ndarray

    :return: None

    """
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

def plot_(history):
    epochs = [i for i in range(20)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(16,9)

    plt.axes(ax[0]).plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
    plt.axes(ax[0]).plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
    plt.axes(ax[0]).set_title('Training & Validation Accuracy')
    plt.axes(ax[0]).legend()
    plt.axes(ax[0]).set_xlabel("Epochs")
    plt.axes(ax[0]).set_ylabel("Accuracy")

    plt.axes(ax[1]).plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
    plt.axes(ax[1]).plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
    plt.axes(ax[1]).set_title('Testing Accuracy & Loss')
    plt.axes(ax[1]).legend()
    plt.axes(ax[1]).set_xlabel("Epochs")
    plt.axes(ax[1]).set_ylabel("Loss")
