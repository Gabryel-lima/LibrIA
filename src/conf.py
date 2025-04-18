import os
import string
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

# Super-Potato
#__DEVICE__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Potato
__DEVICE__ = torch.device("cpu")

# Configuration
class Config_Img_Classifier:
    # Data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data/archive/ASL_Alphabet_Dataset/asl_alphabet_train")
    # LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]
    IMG_SIZE = 32
    BATCH_SIZE = 64
    
    # Model
    # NUM_CLASSES = len(LABELS)
    DROPOUT = 0.5
    
    # Training
    EPOCHS = 10
    LR = 0.001
    MOMENTUM = 0.9
    
    # Paths
    MODEL_DIR = os.path.join(BASE_DIR, "./")
    BEST_MODEL = os.path.join(MODEL_DIR, "asl_vgg_best_weights.keras")
    
    
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

__all__ = [
    '__DEVICE__',
    'Config_Img_Classifier',
]