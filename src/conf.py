import os
import string
import random
import numpy as np
import tensorflow as tf

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

class CFG_VGG16:
    """torch e OpenCV"""
    
    # Diretórios
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data/archive/ASL_Alphabet_Dataset/asl_alphabet_train")

    # Labels (29 classes)
    LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]

    # Imagens
    IMG_SIZE = 224  # compatível com VGG16
    IMG_CHANNELS = 3

    # Hiperparâmetros
    NUM_CLASSES = len(LABELS)
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    MOMENTUM = 0.9
    DROPOUT = 0.5
    
    # Caminhos
    MODEL_DIR = os.path.join(BASE_DIR, "saved", "vgg16")
    PLOTS_DIR = os.path.join(BASE_DIR, "./training_plots")
    MODEL_PATH = "src/saved/vgg16/asl_vgg16_best_weights.h5" #os.path.join(MODEL_DIR, "asl_model.keras") # Modelo treinado a partir do notebook jupyter

    # Semente
    SEED = 42

    @staticmethod
    def seed_everything():
        random.seed(CFG_VGG16.SEED)
        os.environ["PYTHONHASHSEED"] = str(CFG_VGG16.SEED)
        np.random.seed(CFG_VGG16.SEED)
        tf.random.set_seed(CFG_VGG16.SEED)
