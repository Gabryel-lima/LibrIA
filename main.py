from src.utils.imports import (
    np, 
    tf, 
    plt, 
    traceback, 
    json
)
from src.utils.plots import (
    plot_training_history,
    #plot_
    sample_correct_class
)

import keras
from src.core.libria import Libria

# Defina o layer de tradução aleatória fora da função `augment`
random_translation = keras.layers.RandomTranslation(0.2, 0.2)

def augment(image, label) -> tuple:
    """Função para realizar o aumento de dados com mais ruído e entropia"""
    # Flip horizontal aleatório
    image = tf.image.random_flip_left_right(image)
    
    # Ajuste aleatório de brilho e contraste
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.1, upper=1.4)
    
    # Rotação aleatória
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Converter para 3 canais para aplicar saturação e matiz
    image_rgb = tf.image.grayscale_to_rgb(image)
    image_rgb = tf.image.random_saturation(image_rgb, lower=0.6, upper=1.6)
    image_rgb = tf.image.random_hue(image_rgb, max_delta=0.2)
    image = tf.image.rgb_to_grayscale(image_rgb)
    
    # Adicionar ruído gaussiano
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    image = tf.add(image, noise)
    
    # Corte aleatório e redimensionamento
    cropped_size = int(0.9 * image.shape[0])
    image = tf.image.random_crop(image, size=[cropped_size, cropped_size, image.shape[2]])
    
    # Redimensiona novamente para (28, 28, 1)
    image = tf.image.resize(image, [28, 28])  # Redimensiona explicitamente para o tamanho correto
    
    return image, label

def apply_augmentation(image, label) -> tuple:
    """map_function"""
    image = random_translation(image)
    return augment(image, label)

def main():
    try:
        # Inicializar o modelo com a configuração desejada
        libria = Libria(input_shape=(28, 28, 1), num_blocks=[2, 2, 2, 2])
        X_train, X_test, y_train, y_test = libria.load_data()

        libria.ResNet.model.summary()
        libria.ResNet.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Criando o pipeline de dados com tf.data.Dataset e aplicando o aumento de dados
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(64).prefetch(tf.data.AUTOTUNE)

        # Conjunto de validação
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

        # Treinamento do modelo com o tf.data.Dataset
        history = libria.ResNet.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10
        )

        # Plotar e salvar o histórico de treinamento
        plot_training_history(history)

        # Avaliação final
        test_loss, test_accuracy = libria.ResNet.model.evaluate(val_dataset)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

        # Realizar predições no conjunto de teste
        predictions = libria.ResNet.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)  # Pega o índice da maior probabilidade para cada amostra
        true_classes = np.argmax(y_test, axis=1)  # Converte one-hot para índices

        # Identificar previsões corretas
        correct = np.nonzero(predicted_classes == true_classes)[0]

        # Visualizar algumas imagens corretamente classificadas
        plt.figure(figsize=(10, 10))
        for i, c in enumerate(correct[:6]):
            plt.subplot(3, 2, i + 1)
            plt.imshow(X_test[c].reshape(28, 28), cmap='gray', interpolation='none')
            plt.title(f'Pred: {predicted_classes[c]}, True: {true_classes[c]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png')

        # Savar o modelo
        libria.ResNet.model.save('./model/LibriaResNet18.keras')

        # Salvar resultados em um arquivo JSON
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
        # Captura de exceções e salvamento no arquivo de log
        with open('error_log.txt', 'w') as f:
            f.write('An exception occurred:\n')
            f.write(traceback.format_exc())
        print('An exception occurred. Check error_log.txt for details.')

def main_cam():
    import cv2 as cv
    import numpy as np
    from src.core.res_net import ResNet

    # Inicializar o modelo com a configuração desejada
    input_shape = (28, 28, 1)
    num_classes = 25  # Defina o número de classes conforme necessário
    model = ResNet(input_shape=input_shape, num_classes_units=num_classes)

    # Carregar os pesos do modelo salvo
    model.model.load_weights('./model/LibriaResNet18.keras')

    # Dicionário de mapeamento: substitua pelos nomes reais das classes
    class_labels = {
        0: "Sinal A", 1: "Sinal B", 2: "Sinal C", 3: "Sinal D", 4: "Sinal E",
        5: "Sinal F", 6: "Sinal G", 7: "Sinal H", 8: "Sinal I", 9: "Sinal J",
        10: "Sinal K", 11: "Sinal L", 12: "Sinal M", 13: "Sinal N", 14: "Sinal O",
        15: "Sinal P", 16: "Sinal Q", 17: "Sinal R", 18: "Sinal S", 19: "Sinal T",
        20: "Sinal U", 21: "Sinal V", 22: "Sinal W", 23: "Sinal X", 24: "Sinal Y"
    }

    # Configurar a captura de vídeo
    cap = cv.VideoCapture(0)  # Substitua pelo IP do DroidCam

    # Verificar se a captura de vídeo foi inicializada com sucesso
    if not cap.isOpened():
        print("Erro ao abrir a câmera do DroidCam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível receber o frame (stream end?). Saindo ...")
            break

        # Pré-processamento do frame para o formato do modelo
        input_frame = cv.resize(frame, (28, 28))  # Redimensiona para 28x28 pixels
        input_frame = cv.cvtColor(input_frame, cv.COLOR_BGR2GRAY)  # Converte para escala de cinza
        input_frame = input_frame.astype("float32") / 255.0  # Normaliza para [0, 1]
        input_frame = np.expand_dims(input_frame, axis=[0, -1])  # Adiciona batch e canal para compatibilidade com o modelo

        # Fazer a predição
        prediction = model.model.predict(input_frame)
        pred_label = np.argmax(prediction)  # Obtém a classe prevista

        # Obtém a legenda do sinal correspondente
        label_text = class_labels.get(pred_label, "Desconhecido")

        # Exibir o frame com a classe prevista
        cv.putText(frame, f"Classe Prevista: {label_text}, {pred_label}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow("Libria - DroidCam", frame)

        # Pressione 'q' para sair do loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura de vídeo e fechar as janelas
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
