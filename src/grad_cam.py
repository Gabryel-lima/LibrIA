import cv2
import torch
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras._tf_keras.keras.applications.vgg16 import VGG16, preprocess_input
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint

import tensorflow as tf
from conf import Config_Img_Classifier, CFG, plt

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

