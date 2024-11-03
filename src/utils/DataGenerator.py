# from keras._tf_keras.keras.utils import Sequence
# from src.utils.imports import np

# class DataGenerator(Sequence): #TODO: Depois vou ver isso aqui com mais calma
#     def __init__(self, data, labels, batch_size):
#         self.data = data
#         self.labels = labels
#         self.batch_size = batch_size

#     def __len__(self):
#         return int(np.ceil(len(self.data[0]) / self.batch_size))

#     def __getitem__(self, index): *args e **kwargs talvez ...
#         batch_x1 = self.data[0][index * self.batch_size:(index + 1) * self.batch_size]
#         batch_x2 = self.data[1][index * self.batch_size:(index + 1) * self.batch_size]
#         batch_x3 = self.data[2][index * self.batch_size:(index + 1) * self.batch_size]
#         batch_y = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
#         return [batch_x1, batch_x2, batch_x3], batch_y

# def augment_image(image):
#     """
#     Aumenta a imagem aplicando transformações aleatórias.
#     Args:
#     - image: Imagem em escala de cinza (28x28).
#     Retorna:
#     - Uma lista de imagens aumentadas, incluindo a imagem original.
#     """
#     augmented_images = [image]  # Incluir a imagem original

#     # Aumentos de dados
#     for _ in range(2):  # Gerar 4 variações de cada imagem
#         # Ajuste de brilho e contraste aleatório
#         alpha = 1.0 + np.random.uniform(-0.3, 0.3)  # Contraste
#         beta = np.random.uniform(-20, 20)  # Brilho
#         augmented = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

#         # Rotação aleatória
#         angle = np.random.uniform(-15, 15)
#         center = (image.shape[1] // 2, image.shape[0] // 2)
#         matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(augmented, matrix, (image.shape[1], image.shape[0]))

#         # Flip horizontal
#         flipped = cv2.flip(rotated, 1)

#         # Adicionar variações ao conjunto
#         augmented_images.extend([rotated, flipped])

#     return augmented_images