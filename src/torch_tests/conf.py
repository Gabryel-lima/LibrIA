import os
import string
import torch
import random
import numpy as np
import tensorflow as tf

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Dispositivo (usa CUDA se disponível)
__DEVICE__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CFG_HybridASLNet:
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
    BEST_MODEL = os.path.join(BASE_DIR, "./saved/classification_vgg16.pth")
    PLOTS_DIR = os.path.join(BASE_DIR, "./training_plots")

    # Semente
    SEED = 42

    @staticmethod
    def seed_everything():
        random.seed(CFG_HybridASLNet.SEED)
        os.environ["PYTHONHASHSEED"] = str(CFG_HybridASLNet.SEED)
        np.random.seed(CFG_HybridASLNet.SEED)
        tf.random.set_seed(CFG_HybridASLNet.SEED)
        torch.manual_seed(CFG_HybridASLNet.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CFG_HybridASLNet.SEED)

    # Caminhos
    MODEL_DIR = os.path.join(BASE_DIR, "saved")
    BEST_MODEL = os.path.join(MODEL_DIR, "hybrid_aslnet.pth")

class CFG_ASLNet:
    """Tensorflow, keras e OpenCV"""
    
    # Diretórios
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data/archive/ASL_Alphabet_Dataset/asl_alphabet_train")

    # Labels (29 classes)
    LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]

    # Imagens
    IMG_SIZE = 64
    IMG_CHANNELS = 3

    # Hiperparâmetros
    NUM_CLASSES = len(LABELS)
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    MOMENTUM = 0.9
    DROPOUT = 0.5
    
    def seed_everything():
        random.seed(CFG_ASLNet.SEED)
        os.environ["PYTHONHASHSEED"] = str(CFG_ASLNet.SEED)
        np.random.seed(CFG_ASLNet.SEED)
        tf.random.set_seed(CFG_ASLNet.SEED)
        torch.manual_seed(CFG_ASLNet.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CFG_ASLNet.SEED)
            
    # Caminhos
    MODEL_DIR = os.path.join(BASE_DIR, "saved")
    BEST_MODEL = os.path.join(MODEL_DIR, "classification.pth")

__all__ = [
    '__DEVICE__',
    'CFG_HybridASLNet',
]