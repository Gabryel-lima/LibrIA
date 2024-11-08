import tensorflow as tf
import numpy as np
from src.core.MultiModalGestureNet import MultiModalGestureNet

def test_model_inference(model, input_shapes):
    """
    Função para testar a inferência do modelo com as entradas corretamente estruturadas.

    Args:
        model: o modelo Keras a ser testado.
        input_shapes: uma lista de tuplas que representa os formatos das entradas esperadas.
    """
    try:
        # Gerar dados de entrada de exemplo com base nas formas de entrada
        inputs = []
        for shape in input_shapes:
            tensor_input = np.random.random(shape).astype('float32')
            tensor_input = tf.convert_to_tensor(tensor_input)  # Converter para TensorFlow tensor
            inputs.append(tensor_input)

        # Realizar inferência
        predictions = model(inputs)
        print("Inferência realizada com sucesso!")
        print("Saída da inferência:", predictions)

    except Exception as e:
        print(f"Erro ao realizar inferência: {e}")

# Exemplo de uso:
if __name__ == '__main__':
    # Inicializar o modelo MultiModalGestureNet (apenas exemplo)
    libria = MultiModalGestureNet(image_shape=(28, 28, 1), gesture_features_dim=42, num_blocks=[2, 2, 2, 2], num_classes=29)
    libria.build_model()

    # Testar inferência no modelo
    input_shapes = [(1, 28, 28, 1), (1, 42), (1, 28, 28, 1)]  # Formatos das três entradas
    test_model_inference(libria.model, input_shapes)
