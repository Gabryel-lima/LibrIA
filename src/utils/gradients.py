import tensorflow as tf
from keras import Model
from keras.src.losses import CategoricalCrossentropy

def value_gradient(model: Model, inputs, labels) -> None:
    # Converter inputs para tensores do TensorFlow
    inputs = [tf.convert_to_tensor(input_data) for input_data in inputs]

    # Certificar que as variáveis estejam sob observação do tape, tornando-o persistente
    with tf.GradientTape(persistent=True) as tape:
        # Observar as entradas também, além das variáveis treináveis
        for input_tensor in inputs:
            tape.watch(input_tensor)

        # Passar os dados pelo modelo
        outputs = model(inputs, training=True)  # O `training=True` assegura que o tape observe as variáveis em treinamento

        # Calcular a perda entre a previsão (outputs) e os rótulos verdadeiros (labels)
        loss_fn = CategoricalCrossentropy()
        class_out, seg_out, ter_out = outputs
        labels = tf.convert_to_tensor(labels)  # Converter labels para tensor
        loss = loss_fn(labels, class_out)

    # Calcular os gradientes da perda em relação às entradas
    gradients_inputs = tape.gradient(loss, inputs)
    if gradients_inputs is None:
        print("\nOs gradientes em relação às entradas não foram calculados.")
    else:
        # Verificar gradientes das entradas
        for idx, grad in enumerate(gradients_inputs):
            if grad is None:
                print(f"Gradiente não foi calculado para a entrada {idx}")
            elif tf.reduce_sum(grad) == 0:
                print(f"Gradiente zerado encontrado na entrada {idx}")
            else:
                print(f"Gradiente válido encontrado na entrada {idx}")

    # Calcular os gradientes da perda em relação às variáveis treináveis do modelo
    gradients_model = tape.gradient(loss, model.trainable_variables)
    if gradients_model is None:
        print("\nOs gradientes em relação às variáveis do modelo não foram calculados.")
    else:
        # Verificar gradientes do modelo
        for idx, grad in enumerate(gradients_model):
            if grad is None:
                print(f"Gradiente não foi calculado para a variável {model.trainable_variables[idx].name}")
            elif tf.reduce_sum(grad) == 0:
                print(f"Gradiente zerado encontrado na camada {model.trainable_variables[idx].name}")
            else:
                print(f"Gradiente válido encontrado na camada {model.trainable_variables[idx].name}")

    # Como o tape é persistente, você deve liberá-lo manualmente após o uso
    del tape

