import cv2
import numpy as np
import tensorflow as tf
import string
import matplotlib.pyplot as plt

class GradCAM:
    # https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
            
    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
            
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])
        
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
            
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        
        # return the resulting heatmap to the calling function
        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):
        
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
    
    def gradcam_images(self, train_dir, test_dir, labels, fine_tuned_model):
        # Defina a transformação e carregamento de dados
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        
        # Crie um gerador para o diretório de teste
        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(CFG.img_height, CFG.img_width),
            batch_size=1,  # Use 1 para processar uma imagem de cada vez
            class_mode='categorical', 
            shuffle=False
        )

        # Crie subplots
        fig, axs = plt.subplots(len(labels), 7, figsize=(12, 10))

        for i, label in enumerate(labels):
            axs[i, 0].text(0.5, 0.5, label, ha='center', va='center', fontsize=8)
            axs[i, 0].axis('off')

            # Escolha uma imagem de cada vez para a geração de GradCAM
            image, _ = test_generator.next()

            # Faça a previsão e obtenha a classe
            img_label_ci = fine_tuned_model.predict(image, verbose=0)
            img_label = np.argmax(img_label_ci[0])

            # Calcule o Grad-CAM
            cam = GradCAM(fine_tuned_model, img_label)
            heatmap = cam.compute_heatmap(image[0])

            # Sobreponha o mapa de calor na imagem original
            heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            (heatmap, output) = cam.overlay_heatmap(heatmap, image[0], alpha=0.5)

            # Exiba a imagem original, mapa de calor e sobreposição
            axs[i, 1].imshow(image[0])
            axs[i, 1].axis("off")
            axs[i, 2].imshow(heatmap)
            axs[i, 2].axis("off")
            axs[i, 3].imshow(output)
            axs[i, 3].axis("off")

        # Título e exibição
        plt.suptitle("Class Activation Maps (GradCAM) in Test Images", x=0.55, y=0.92)
        plt.show()

# Configuração
class CFG:
    batch_size = 64
    img_height = 64
    img_width = 64
    epochs = 6
    num_classes = 29
    img_channels = 3

    TRAIN_PATH = "./data/archive/ASL_Alphabet_Dataset/asl_alphabet_train"
    TEST_PATH = "./data/archive/ASL_Alphabet_Dataset/asl_alphabet_test"
    # LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]
    
    labels = []
    alphabet = list(string.ascii_uppercase)
    labels.extend(alphabet)
    labels.extend(["del", "nothing", "space"])
    print(labels)

    def seed_everything(seed=2023):
        import os
        import random
        import numpy as np
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

"""Api keras legacy"""
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras._tf_keras.keras.applications.vgg16 import VGG16, preprocess_input
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint

# Configuração
MODEL_WEIGHTS_PATH = "src/saved/asl_vgg16_best_weights.keras"

def open_camera(ip_url="http://192.168.1.3:4747/video", fallback_device=0):
    """Tenta abrir a câmera IP; se falhar, tenta abrir a câmera local."""
    cap = cv2.VideoCapture(ip_url)
    if cap.isOpened():
        print("[INFO] Câmera IP conectada com sucesso.")
        return cap
    else:
        print("[WARN] Falha ao conectar à câmera IP. Tentando câmera local...")
        cap.release()
        cap = cv2.VideoCapture(fallback_device)
        if cap.isOpened():
            print("[INFO] Câmera local conectada com sucesso.")
            return cap
        else:
            print("[ERROR] Nenhuma câmera disponível!")
            return None

def preprocess_frame_ImageNet(frame, img_size=224):
    """ Converte o frame para RGB, redimensiona e normaliza para a ImageNet """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte para RGB
    resized_frame = cv2.resize(frame_rgb, (img_size, img_size))  # Redimensiona para 224x224
    img_array = img_to_array(resized_frame)  # Converte para array
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona batch dimension
    img_array = preprocess_input(img_array)  # Pré-processamento específico para ImageNet
    return img_array, resized_frame

def generate_gradcam(model, image_tensor, target_layer_name="block5_conv3"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / tf.math.reduce_max(cam)
    cam = tf.image.resize(cam[..., tf.newaxis], (224, 224)).numpy()
    return cam.squeeze()

# Configuração de modelo
def create_model():
    """ Cria o modelo VGG16 com camadas congeladas e novas camadas para classificação """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layer in base_model.layers:
        layer.trainable = False  # Congelar as camadas do modelo pré-treinado

    # Adicionar camadas customizadas para classificação
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(29, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(MODEL_WEIGHTS_PATH)  # Carregar pesos salvos
    model.evaluate()  # Colocar o modelo em modo de avaliação
    model.summary()
    return model

def camImageNet():
    """ Aplica Grad-CAM usando o modelo VGG16 """
    try:
        model = create_model()  # Cria o modelo com o VGG16

        # Defina a camada alvo para o Grad-CAM (última camada convolucional de VGG16)
        target_layer = model.get_layer("block5_conv3")
        cam_generator = GradCAM(model, target_layer)

    except Exception as e:
        print(f"[ERROR] Falha ao carregar o modelo: {e}")
        return

    # Tenta abrir a câmera
    cap = open_camera()
    if cap is None:
        return

    window_name = "Câmera + Inferência + Grad-CAM"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[WARN] Frame inválido recebido. Tentando novamente...")
            continue

        # Pré-processamento do frame
        input_tensor, gray_frame = preprocess_frame_ImageNet(frame, 224)  # Para o VGG16
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        # Inferência
        try:
            output = model(input_tensor, training=False)
            predicted_label = tf.argmax(output[0]).numpy()
            predicted_name = CFG.labels[predicted_label]
        except Exception as e:
            print(f"[ERROR] Falha na inferência: {e}")
            predicted_name = "Erro"
            continue  # Pula este frame em caso de erro

        # Gera o Grad-CAM
        cam_map = generate_gradcam(model, input_tensor)
        heatmap = np.uint8(255 * cam_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        combined = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Combina a imagem original com o mapa de calor
        combined = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Overlay de texto com a previsão na imagem combinada
        cv2.putText(combined, f"Pred: {predicted_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar a imagem combinada
        cv2.imshow(window_name, combined)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC para sair
            print("[INFO] Encerrando...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camImageNet()
    