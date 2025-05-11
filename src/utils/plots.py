from src.utils.imports import (
    plt # from matplotlib.pyplot as plt 
)

def plot_training_history(history, save_path='training_history.png'):
    """Função para plotar o histórico de treinamento e validação.
    
    Args:
        history: Histórico de treinamento retornado pelo método model.fit().
        save_path: Caminho para salvar a imagem do gráfico.
    """
    # Recuperando métricas do histórico
    loss = history['training_loss']
    val_loss = history['validation_loss']
    accuracy = history['training_accuracy']
    val_accuracy = history['validation_accuracy']
    
    # Criar o gráfico com 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot da perda
    axs[0].plot(loss, label='Training Loss', color='blue')
    axs[0].plot(val_loss, label='Validation Loss', color='orange')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    # Subplot da precisão
    axs[1].plot(accuracy, label='Training Accuracy', color='blue')
    axs[1].plot(val_accuracy, label='Validation Accuracy', color='orange')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Ajustar espaçamento e salvar o gráfico
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def sample_correct_class():
    import numpy as np
    import matplotlib.pyplot as plt
    from src.torch_tests.ASLnet import Libria  # Certifique-se de que Libria está configurado corretamente

    # Inicialização da rede e carregamento de dados
    libria = Libria(input_shape=(28, 28, 1), num_blocks=[2, 2, 2, 2])
    X_train, _, y_train, _ = libria.load_data()  # Carregar dados de treino e rótulos

    # Carregar pesos do modelo treinado
    libria.model.load_weights('./model/best_model.keras')  # Substitua pelo caminho correto do arquivo de pesos

    # Obter previsões no conjunto de treino
    predictions = libria.model.predict(X_train)
    predicted_classes = np.argmax(predictions, axis=-1)  # Classe prevista para cada amostra
    true_classes = np.argmax(y_train, axis=-1)  # Classe real para cada amostra

    # Criação de uma amostra por classe
    num_classes = len(np.unique(true_classes))
    sample_per_class = {}

    for i, (image, true_class, pred_class) in enumerate(zip(X_train, true_classes, predicted_classes)):
        # Salvar a primeira imagem de cada classe
        if true_class not in sample_per_class:
            sample_per_class[true_class] = (image, true_class, pred_class)
        # Parar após encontrar uma amostra para cada classe
        if len(sample_per_class) == num_classes:
            break

    # Definir o tamanho do grid dinamicamente
    rows = (num_classes // 6) + 1 if num_classes % 6 != 0 else num_classes // 6
    fig, axes = plt.subplots(rows, 6, figsize=(15, rows * 3))

    # Configurar o layout dos plots
    axes = axes.flatten() if rows > 1 else [axes]
    for ax, (true_class, (image, true, pred)) in zip(axes, sample_per_class.items()):
        ax.imshow(image.reshape(28, 28), cmap="gray")
        ax.axis("off")
        # Definir título com classe prevista e verdadeira
        ax.set_title(f"Real: {true}, Pred: {pred}", color=("green" if true == pred else "red"))

    plt.suptitle("Amostra de Sinais de Mão Classificados", fontsize=16)
    plt.tight_layout()
    plt.show()
